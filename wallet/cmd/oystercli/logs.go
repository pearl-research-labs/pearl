// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// logsScreen inspects the local oyster log file. It reads straight from
// disk, so it also works while the daemon is down.
func logsScreen(cfg *config) error {
	path := cfg.logFilePath()

	for {
		const (
			opTail   = "tail"
			opMore   = "more"
			opFollow = "follow"
			opPath   = "path"
			opBack   = "back"
		)
		choice := opTail // first option, where huh starts the cursor
		submitted, err := runForm(newForm(huh.NewGroup(
			huh.NewSelect[string]().
				Title("Logs").
				Description(path).
				Options(
					huh.NewOption("Show the last 40 lines", opTail),
					huh.NewOption("Show the last 200 lines", opMore),
					huh.NewOption("Follow live (enter stops)", opFollow),
					huh.NewOption("Use a different log file", opPath),
					huh.NewOption("Back", opBack),
				).
				Value(&choice),
		)))
		if err != nil {
			return err
		}
		if !submitted || choice == opBack {
			return nil
		}

		switch choice {
		case opTail:
			printLogTail(path, 40)
		case opMore:
			printLogTail(path, 200)
		case opFollow:
			followLog(path)
		case opPath:
			newPath := path
			ok, err := runForm(newForm(huh.NewGroup(
				huh.NewInput().
					Title("Log file path").
					Validate(func(s string) error {
						if !fileExists(cleanAndExpandPath(strings.TrimSpace(s))) {
							return fmt.Errorf("file does not exist")
						}
						return nil
					}).
					Value(&newPath),
			)))
			if err != nil {
				return err
			}
			if ok {
				path = cleanAndExpandPath(strings.TrimSpace(newPath))
			}
		}
	}
}

// printLogTail prints the last n lines of the file with level colorization.
func printLogTail(path string, n int) {
	lines, err := tailLines(path, n)
	if err != nil {
		printError(err)
		return
	}
	if len(lines) == 0 {
		printWarn("Log file is empty.")
		return
	}
	printTitle(fmt.Sprintf("%s — last %d lines", path, len(lines)))
	for _, line := range lines {
		lipgloss.Println(colorizeLogLine(line))
	}
}

// tailLines returns up to n trailing lines. Oyster caps log files at ~10MB
// before rotation, so reading the file whole is acceptable.
func tailLines(path string, n int) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines, nil
}

// followLog streams appended lines until the user presses enter.
func followLog(path string) {
	fi, err := os.Stat(path)
	if err != nil {
		printError(err)
		return
	}
	offset := fi.Size()

	lipgloss.Println(th.subtle.Render("Following " + path + " — press enter to stop."))

	stop := make(chan struct{})
	go func() {
		reader := bufio.NewReader(os.Stdin)
		_, _ = reader.ReadString('\n')
		close(stop)
	}()

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			offset = printLogDelta(path, offset)
		}
	}
}

// printLogDelta prints anything appended past offset and returns the new
// offset. Truncation (rotation) restarts from the beginning.
func printLogDelta(path string, offset int64) int64 {
	f, err := os.Open(path)
	if err != nil {
		return offset
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return offset
	}
	if fi.Size() < offset {
		offset = 0
	}
	if fi.Size() == offset {
		return offset
	}
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return offset
	}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)
	for scanner.Scan() {
		lipgloss.Println(colorizeLogLine(scanner.Text()))
	}
	return fi.Size()
}

// colorizeLogLine highlights btclog level tags. The line is sanitized first:
// logs contain remote-controlled strings (e.g. peer user agents, which the
// wire protocol only length-limits), so raw control characters would let a
// peer inject terminal escapes — prompt spoofing, OSC clipboard writes —
// into the viewer.
func colorizeLogLine(line string) string {
	line = stripControlRunes(line)
	switch {
	case strings.Contains(line, "[ERR]"), strings.Contains(line, "[CRT]"):
		return th.bad.Render(line)
	case strings.Contains(line, "[WRN]"):
		return th.warn.Render(line)
	case strings.Contains(line, "[DBG]"), strings.Contains(line, "[TRC]"):
		return th.subtle.Render(line)
	default:
		return th.value.Render(line)
	}
}

// stripControlRunes replaces C0/C1 control characters (except tab) with the
// Unicode replacement character, keeping injected sequences visible rather
// than silently dropping them. ansi.Strip is deliberately not used here: it
// removes well-formed escape sequences but passes bare C0 controls through
// (its Execute action keeps them), so carriage-return and backspace spoofing
// would survive it.
func stripControlRunes(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '\t' {
			return r
		}
		if r < 0x20 || r == 0x7f || (r >= 0x80 && r <= 0x9f) {
			return '\uFFFD'
		}
		return r
	}, s)
}
