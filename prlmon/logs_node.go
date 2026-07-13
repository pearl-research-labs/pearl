package main

// Handlers for pearld's log endpoints under /node/logs, plus shared query
// parsing used by both /node/logs and /logs (sidecar self):
//
//   GET /node/logs                    stream the active log (cat / head / tail / follow)
//   GET /node/logs/files              list the active log + rotated archives
//   GET /node/logs/files/<name>       download a specific file verbatim

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Shared query parsing (consumed by both /node/logs and /logs handlers)
// ---------------------------------------------------------------------------

// logQuery is the parsed view of /node/logs and /logs query parameters.
//
// Modes (mutually exclusive shape, validated in parseLogQuery):
//
//   - Tail==0 && Head==0 && !Follow → return the full file content
//   - Head>0                        → return first Head lines
//   - Tail>0                        → return last Tail lines
//   - Follow                        → stream from the current end
//   - Tail>0 && Follow              → backfill last Tail lines, then stream
type logQuery struct {
	Tail   int
	Head   int
	Follow bool
}

func parseLogQuery(r *http.Request, maxLines int) (logQuery, error) {
	var q logQuery

	tailStr := r.URL.Query().Get("tail")
	headStr := r.URL.Query().Get("head")
	if tailStr != "" && headStr != "" {
		return q, fmt.Errorf("tail and head are mutually exclusive")
	}

	if tailStr != "" {
		n, err := parseLineCount(tailStr, maxLines)
		if err != nil {
			return q, fmt.Errorf("invalid tail=%q: %v", tailStr, err)
		}
		q.Tail = n
	}
	if headStr != "" {
		n, err := parseLineCount(headStr, maxLines)
		if err != nil {
			return q, fmt.Errorf("invalid head=%q: %v", headStr, err)
		}
		q.Head = n
	}

	switch r.URL.Query().Get("follow") {
	case "", "false", "0":
		q.Follow = false
	case "true", "1":
		q.Follow = true
	default:
		return q, fmt.Errorf("invalid follow: must be true or false")
	}
	if q.Follow && q.Head > 0 {
		return q, fmt.Errorf("follow=true cannot be combined with head")
	}

	return q, nil
}

// parseLineCount accepts a positive integer, clamped to [1, maxLines].
func parseLineCount(v string, maxLines int) (int, error) {
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("must be positive integer")
	}
	if n > maxLines {
		n = maxLines
	}
	return n, nil
}

// ---------------------------------------------------------------------------
// /node/logs
// ---------------------------------------------------------------------------

// tailChunkSize is the read window when scanning a log file backwards. Big
// enough to capture many lines per syscall, small enough to bound memory when
// callers ask for a few lines from a huge file.
const tailChunkSize = 64 * 1024

// followPollInterval is how often we re-stat the followed file to detect new
// data and rotation.
const followPollInterval = 500 * time.Millisecond

// filesPrefix is the URL prefix under which the active log file and its
// rotated archives are downloaded. Anything below it is treated as a
// filename to resolve against the configured log directory.
const filesPrefix = "/node/logs/files/"

// archiveSuffixRE matches `<N>` or `<N>.gz` (case-sensitive, no leading dot).
var archiveSuffixRE = regexp.MustCompile(`^([0-9]+)(?:\.gz)?$`)

// ---------------------------------------------------------------------------
// /node/logs
// ---------------------------------------------------------------------------

func (m *Monitor) handleLogsFile(w http.ResponseWriter, r *http.Request) {
	if m.cfg.NodeLogFile == "" {
		http.Error(w, "/node/logs disabled: set --node-log-file to enable", http.StatusNotFound)
		return
	}

	q, err := parseLogQuery(r, m.cfg.LogsMaxLines)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")

	switch {
	case q.Follow:
		m.streamLogsFile(w, r, q)
	case q.Head > 0:
		writeFirstLines(w, m.cfg.NodeLogFile, q.Head)
	case q.Tail > 0:
		writeLastLines(w, m.cfg.NodeLogFile, q.Tail)
	default:
		writeFullFile(w, m.cfg.NodeLogFile)
	}
}

// writeFullFile streams the active log file's contents verbatim, like `cat`.
func writeFullFile(w http.ResponseWriter, path string) {
	f, err := os.Open(path)
	if err != nil {
		http.Error(w, "failed to open log: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()
	if _, err := io.Copy(w, f); err != nil {
		log.Warnf("logs full-stream error: %v", err)
	}
}

// writeFirstLines writes the first N lines of path to w.
func writeFirstLines(w http.ResponseWriter, path string, n int) {
	f, err := os.Open(path)
	if err != nil {
		http.Error(w, "failed to open log: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for i := 0; i < n && scanner.Scan(); i++ {
		if _, err := fmt.Fprintln(w, scanner.Text()); err != nil {
			return
		}
	}
}

// writeLastLines writes the last N lines of path to w in chronological order.
//
// Reads backwards in fixed chunks so big logs don't have to be fully scanned.
func writeLastLines(w http.ResponseWriter, path string, n int) {
	lines, err := readLastLines(path, n)
	if err != nil {
		http.Error(w, "failed to read log: "+err.Error(), http.StatusInternalServerError)
		return
	}
	for _, l := range lines {
		if _, err := fmt.Fprintln(w, l); err != nil {
			return
		}
	}
}

// readLastLines returns up to maxLines from the end of path, in chronological
// order (oldest first).
func readLastLines(path string, maxLines int) ([]string, error) {
	if maxLines <= 0 {
		return nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	out := make([]string, 0, maxLines)
	// remainder holds bytes from the *start* of a chunk that didn't end at a
	// newline — they belong to the line whose tail is in the *previous*
	// (later-in-file) chunk.
	var remainder []byte
	pos := stat.Size()
	for pos > 0 && len(out) < maxLines {
		readSize := int64(tailChunkSize)
		if readSize > pos {
			readSize = pos
		}
		pos -= readSize
		buf := make([]byte, readSize)
		if _, err := f.ReadAt(buf, pos); err != nil && err != io.EOF {
			return nil, err
		}
		if len(remainder) > 0 {
			buf = append(buf, remainder...)
			remainder = nil
		}

		// Walk backwards through buf, emitting one line per newline.
		end := len(buf)
		for i := len(buf) - 1; i >= 0; i-- {
			if buf[i] != '\n' {
				continue
			}
			if i+1 <= end {
				line := string(buf[i+1 : end])
				if line != "" {
					out = append(out, line)
					if len(out) >= maxLines {
						break
					}
				}
			}
			end = i
		}
		// buf[0:end] is a partial line. Either it's the very first line of
		// the file (when pos==0) or it's the prefix of a line whose suffix
		// lives in the next-earlier chunk.
		if end > 0 && len(out) < maxLines {
			if pos == 0 {
				line := string(buf[:end])
				if line != "" {
					out = append(out, line)
				}
			} else {
				remainder = append([]byte(nil), buf[:end]...)
			}
		}
	}

	// out is currently newest-first; flip to chronological.
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out, nil
}

// streamLogsFile is the follow=true path. Optionally backfills q.Tail lines
// for context, then watches the file for new content (re-opening on rotation)
// until the client disconnects.
func (m *Monitor) streamLogsFile(w http.ResponseWriter, r *http.Request, q logQuery) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported by this server", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusOK)
	// Flush response headers immediately so the client can start reading
	// even before the first new line arrives (otherwise idle follow requests
	// look like they hang).
	flusher.Flush()

	if q.Tail > 0 {
		if hist, err := readLastLines(m.cfg.NodeLogFile, q.Tail); err == nil {
			for _, l := range hist {
				if _, err := fmt.Fprintln(w, l); err != nil {
					return
				}
			}
			flusher.Flush()
		}
	}

	file, fi, err := openAndStat(m.cfg.NodeLogFile)
	if err != nil {
		fmt.Fprintf(w, "# follow error: %s\n", err.Error())
		flusher.Flush()
		return
	}
	defer file.Close()
	if _, err := file.Seek(0, io.SeekEnd); err != nil {
		return
	}

	reader := bufio.NewReader(file)
	poll := time.NewTicker(followPollInterval)
	defer poll.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-poll.C:
			rotated, err := isRotated(m.cfg.NodeLogFile, fi)
			if err == nil && rotated {
				if newFile, newFI, err := openAndStat(m.cfg.NodeLogFile); err == nil {
					file.Close()
					file = newFile
					fi = newFI
					reader = bufio.NewReader(file)
				}
			}
			any := false
			for {
				line, err := reader.ReadString('\n')
				if len(line) > 0 {
					if _, werr := fmt.Fprint(w, line); werr != nil {
						return
					}
					any = true
				}
				if err != nil {
					if err != io.EOF {
						return
					}
					break
				}
			}
			if any {
				flusher.Flush()
			}
		}
	}
}

func openAndStat(path string) (*os.File, os.FileInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, nil, err
	}
	return file, stat, nil
}

// isRotated re-stats path and compares with prev using os.SameFile (which
// checks inode+device on unix). If they differ the file was rotated.
func isRotated(path string, prev os.FileInfo) (bool, error) {
	cur, err := os.Stat(path)
	if err != nil {
		return false, err
	}
	return !os.SameFile(prev, cur), nil
}

// ---------------------------------------------------------------------------
// /node/logs/files
// ---------------------------------------------------------------------------

// logFile is one entry returned by GET /node/logs/files.
type logFile struct {
	Name       string    `json:"name"`
	Active     bool      `json:"active"`
	SizeBytes  int64     `json:"sizeBytes"`
	ModifiedAt time.Time `json:"modifiedAt"`
	Compressed bool      `json:"compressed"`
}

func (m *Monitor) handleLogFilesList(w http.ResponseWriter, r *http.Request) {
	if m.cfg.NodeLogFile == "" {
		http.Error(w, "/node/logs/files disabled: set --node-log-file to enable", http.StatusNotFound)
		return
	}

	files, err := listLogFiles(m.cfg.NodeLogFile)
	if err != nil {
		http.Error(w, "failed to list log files: "+err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, files)
}

func (m *Monitor) handleLogFileDownload(w http.ResponseWriter, r *http.Request) {
	if m.cfg.NodeLogFile == "" {
		http.Error(w, "/node/logs/files disabled: set --node-log-file to enable", http.StatusNotFound)
		return
	}

	name := strings.TrimPrefix(r.URL.Path, filesPrefix)
	if name == "" {
		http.Error(w, "missing file name", http.StatusBadRequest)
		return
	}
	// Reject any path separator or traversal attempt outright. Downloads
	// are siblings of the configured log file; we never resolve names that
	// could escape that directory.
	if strings.ContainsAny(name, "/\\") || name == "." || name == ".." {
		http.Error(w, "invalid file name", http.StatusBadRequest)
		return
	}
	if !isLogFileName(m.cfg.NodeLogFile, name) {
		http.NotFound(w, r)
		return
	}

	full := filepath.Join(filepath.Dir(m.cfg.NodeLogFile), name)
	f, err := os.Open(full)
	if err != nil {
		if os.IsNotExist(err) {
			http.NotFound(w, r)
			return
		}
		http.Error(w, "failed to open file: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		http.Error(w, "failed to stat file: "+err.Error(), http.StatusInternalServerError)
		return
	}

	if strings.HasSuffix(name, ".gz") {
		w.Header().Set("Content-Type", "application/gzip")
	} else {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	}
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename=%q`, name))
	w.Header().Set("Content-Length", strconv.FormatInt(fi.Size(), 10))
	w.Header().Set("Last-Modified", fi.ModTime().UTC().Format(http.TimeFormat))

	if _, err := io.Copy(w, f); err != nil {
		log.Warnf("file download write error: %v", err)
	}
}

// listLogFiles enumerates the active log file plus rotated siblings. The
// active file is first; rotated archives follow newest-first by index.
func listLogFiles(activePath string) ([]logFile, error) {
	out := make([]logFile, 0, 8)

	if fi, err := os.Stat(activePath); err == nil {
		out = append(out, logFile{
			Name:       filepath.Base(activePath),
			Active:     true,
			SizeBytes:  fi.Size(),
			ModifiedAt: fi.ModTime().UTC(),
		})
	}

	matches, err := filepath.Glob(activePath + ".*")
	if err != nil {
		return nil, err
	}
	type indexed struct {
		entry logFile
		index int
	}
	rotated := make([]indexed, 0, len(matches))
	for _, m := range matches {
		suffix := strings.TrimPrefix(m, activePath+".")
		grp := archiveSuffixRE.FindStringSubmatch(suffix)
		if grp == nil {
			continue
		}
		idx, _ := strconv.Atoi(grp[1])
		fi, err := os.Stat(m)
		if err != nil {
			continue
		}
		rotated = append(rotated, indexed{
			entry: logFile{
				Name:       filepath.Base(m),
				SizeBytes:  fi.Size(),
				ModifiedAt: fi.ModTime().UTC(),
				Compressed: strings.HasSuffix(m, ".gz"),
			},
			index: idx,
		})
	}
	sort.Slice(rotated, func(i, j int) bool { return rotated[i].index > rotated[j].index })
	for _, r := range rotated {
		out = append(out, r.entry)
	}
	return out, nil
}

// isLogFileName reports whether `name` is the active log's basename or a
// valid rotated sibling (`<base>.<N>` or `<base>.<N>.gz`).
func isLogFileName(activePath, name string) bool {
	base := filepath.Base(activePath)
	if name == base {
		return true
	}
	if !strings.HasPrefix(name, base+".") {
		return false
	}
	return archiveSuffixRE.MatchString(strings.TrimPrefix(name, base+"."))
}
