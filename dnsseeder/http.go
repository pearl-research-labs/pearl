package main

import (
	"fmt"
	"html"
	"html/template"
	"log"
	"net/http"
	"time"
)

// startHTTP runs in a goroutine and provides the web interface
// to the dnsseeder
func startHTTP(port string) {

	http.HandleFunc("/dns", dnsWebHandler)
	http.HandleFunc("/node", nodeHandler)
	http.HandleFunc("/statusRG", statusRGHandler)
	http.HandleFunc("/statusCG", statusCGHandler)
	http.HandleFunc("/statusWG", statusWGHandler)
	http.HandleFunc("/statusNG", statusNGHandler)
	http.HandleFunc("/summary", summaryHandler)
	http.HandleFunc("/seeds.txt", txtHandler)
	http.HandleFunc("/", emptyHandler)
	// listen only on localhost
	err := http.ListenAndServe("0.0.0.0:"+port, nil)
	if err != nil {
		log.Fatal("ListenAndServe: ", err)
	}

}

// dnsWebHandler processes all requests and returns output in the requested format
func dnsWebHandler(w http.ResponseWriter, r *http.Request) {

	st := time.Now()

	// skip the s= from the raw query
	n := r.FormValue("s")
	s := getSeederByName(n)
	if s == nil {
		writeHeader(w, r)
		fmt.Fprintf(w, "No seeder found: %s", html.EscapeString(n))
		writeFooter(w, r, st)
		return
	}

	// FIXME - This is ugly code and needs to be cleaned up a lot

	config.dnsmtx.RLock()
	// if the dns map does not have a key for the request it will return an empty slice
	v4std := config.dns[s.dnsHost+".A"]
	v4non := config.dns["nonstd."+s.dnsHost+".A"]
	v6std := config.dns[s.dnsHost+".AAAA"]
	v6non := config.dns["nonstd."+s.dnsHost+".AAAA"]
	config.dnsmtx.RUnlock()

	var v4stdstr, v4nonstr []string
	var v6stdstr, v6nonstr []string

	if x := len(v4std); x > 0 {
		v4stdstr = make([]string, x)
		for k, v := range v4std {
			v4stdstr[k] = v.String()
		}
	} else {
		v4stdstr = []string{"No records Available"}
	}

	if x := len(v4non); x > 0 {
		v4nonstr = make([]string, x)
		for k, v := range v4non {
			v4nonstr[k] = v.String()
		}
	} else {
		v4nonstr = []string{"No records Available"}
	}

	// ipv6
	if x := len(v6std); x > 0 {
		v6stdstr = make([]string, x)
		for k, v := range v6std {
			v6stdstr[k] = v.String()
		}
	} else {
		v6stdstr = []string{"No records Available"}
	}

	if x := len(v6non); x > 0 {
		v6nonstr = make([]string, x)
		for k, v := range v6non {
			v6nonstr[k] = v.String()
		}
	} else {
		v6nonstr = []string{"No records Available"}
	}

	t1 := `
	<center>
	<table border=1>
	  <tr>
	  <th>Standard Ports</th>
	  <th>Non Standard Ports</th>
	  </tr>
	  <tr>
	  <td>
	  `
	t2 := `  {{range .}}
	       {{.}}<br>
	     {{end}}
		 `
	t3 := `
	  </td>
	  <td>
	  `
	t4 := `
	  </td>
	  </tr>
	</table>
	</center>
	`

	writeHeader(w, r)
	fmt.Fprintf(w, "<b>Currently serving the following DNS records</b>")
	fmt.Fprintf(w, "<p><center><b>IPv4</b></center></p>")
	fmt.Fprintf(w, "%s", t1)

	t := template.New("v4 template")
	t, err := t.Parse(t2)
	if err != nil {
		log.Printf("error parsing template v4 %v\n", err)
	}
	err = t.Execute(w, v4stdstr)
	if err != nil {
		log.Printf("error executing template v4 %v\n", err)
	}

	fmt.Fprintf(w, "%s", t3)

	err = t.Execute(w, v4nonstr)
	if err != nil {
		log.Printf("error executing template v4 non %v\n", err)
	}

	fmt.Fprintf(w, "%s", t4)

	// ipv6 records

	fmt.Fprintf(w, "<p><center><b>IPv6</b></center></p>")
	fmt.Fprintf(w, "%s", t1)

	err = t.Execute(w, v6stdstr)
	if err != nil {
		log.Printf("error executing template v6 %v\n", err)
	}

	fmt.Fprintf(w, "%s", t3)

	err = t.Execute(w, v6nonstr)
	if err != nil {
		log.Printf("error executing template v6 non %v\n", err)
	}

	fmt.Fprintf(w, "%s", t4)
	writeFooter(w, r, st)
}

// emptyHandler processes all requests for non-existant urls
func emptyHandler(w http.ResponseWriter, r *http.Request) {

	w.Header().Set("Content-Type", "text/plain")
	fmt.Fprintf(w, "Nothing to see here. Move along please\n")
}

func statusRGHandler(w http.ResponseWriter, r *http.Request) {
	statusHandler(w, r, statusRG)
}
func statusCGHandler(w http.ResponseWriter, r *http.Request) {
	statusHandler(w, r, statusCG)
}
func statusWGHandler(w http.ResponseWriter, r *http.Request) {
	statusHandler(w, r, statusWG)
}
func statusNGHandler(w http.ResponseWriter, r *http.Request) {
	statusHandler(w, r, statusNG)
}

type webstatus struct {
	Key string
	// Value carries the summary cell's intentional <b> markup, so it is
	// template.HTML; generateWebStatus escapes every dynamic string it
	// interpolates (above all the peer-controlled UserAgent) first.
	Value  template.HTML
	Seeder string
}

func statusHandler(w http.ResponseWriter, r *http.Request, status uint32) {

	startT := time.Now()

	// read the seeder name
	n := r.FormValue("s")
	s := getSeederByName(n)
	if s == nil {
		writeHeader(w, r)
		fmt.Fprintf(w, "No seeder found called %s", html.EscapeString(n))
		writeFooter(w, r, startT)
		return
	}

	// gather all the info before writing anything to the remote browser
	ws := generateWebStatus(s, status)

	st := `
	<center>
	<table border=1>
	  <tr>
	  <th>Node</th>
	  <th>Summary</th>
	  </tr>
	     {{range .}}
	  <tr>
	  <td>
	       <a href="/node?s={{.Seeder}}&nd={{.Key}}">{{.Key}}</a>
	  </td>
	  <td>
	       {{.Value}}
	  </td>
	  </tr>
	     {{end}}
	</table>
	</center>
	`

	writeHeader(w, r)

	if len(ws) == 0 {
		fmt.Fprintf(w, "No Nodes found with this status")
	} else {

		switch status {
		case statusRG:
			fmt.Fprintf(w, "<center><b>Node Status: statusRG - (Reported Good) Have not been able to get addresses yet</b></center>")
		case statusCG:
			fmt.Fprintf(w, "<center><b>Node Status: statusCG - (Currently Good) Able to connect and get addresses</b></center>")
		case statusWG:
			fmt.Fprintf(w, "<center><b>Node Status: statusWG - (Was Good) Was Ok but now can not get addresses</b></center>")
		case statusNG:
			fmt.Fprintf(w, "<center><b>Node Status: statusNG - (No Good) Unable to get addresses</b></center>")
		}
		t := template.New("Status template")
		t, err := t.Parse(st)
		if err != nil {
			log.Printf("error parsing status template %v\n", err)
		}
		err = t.Execute(w, ws)
		if err != nil {
			log.Printf("error executing status template %v\n", err)
		}

	}
	writeFooter(w, r, startT)
}

// generateWebStatus is given a node status and returns a slice of webstatus structures
// ready to be ranged over by an html/template
func generateWebStatus(s *dnsseeder, status uint32) (ws []webstatus) {

	s.mtx.RLock()
	defer s.mtx.RUnlock()

	var valueStr string

	for k, v := range s.theList {
		if v.status != status {
			continue
		}

		switch status {
		case statusRG:
			valueStr = fmt.Sprintf("<b>Fail Count:</b> %v <b>DNS Type:</b> %s",
				v.connectFails,
				html.EscapeString(v.dns2str()))
		case statusCG:
			valueStr = fmt.Sprintf("<b>Remote Version:</b> %v%s <b>Last Block:</b> %v <b>DNS Type:</b> %s",
				v.version,
				html.EscapeString(v.strVersion),
				v.lastBlock,
				html.EscapeString(v.dns2str()))

		case statusWG:
			valueStr = fmt.Sprintf("<b>Last Try:</b> %s ago <b>Last Status:</b> %s\n",
				time.Since(v.lastTry).String(),
				html.EscapeString(v.statusStr))

		case statusNG:
			valueStr = fmt.Sprintf("<b>Fail Count:</b> %v <b>Last Try:</b> %s ago <b>Last Status:</b> %s\n",
				v.connectFails,
				time.Since(v.lastTry).String(),
				html.EscapeString(v.statusStr))

		default:
			valueStr = ""
		}

		ows := webstatus{
			Key:    k,
			Value:  template.HTML(valueStr),
			Seeder: s.name,
		}
		ws = append(ws, ows)
	}

	return ws
}

// copy Node details into a template friendly struct
type webtemplate struct {
	Key            string
	IP             string
	Port           uint16
	Statusstr      string
	Rating         string
	Dnstype        string
	Lastconnect    string
	Lastconnectago string
	Lasttry        string
	Lasttryago     string
	Crawlstart     string
	Crawlstartago  string
	Crawlactive    bool
	Connectfails   uint32
	Version        int32
	Strversion     string
	Services       string
	Lastblock      int32
	Nonstdip       string
}

// nodeHandler displays details about one node
func nodeHandler(w http.ResponseWriter, r *http.Request) {

	st := time.Now()

	ndt := `
    <center>
    <table border=1>
      <tr>
      <th>Node {{.Key}}</th><th>Details</th>
      </tr>
      <tr><td>IP Address</td><td>{{.IP}}</td></tr>
      <tr><td>Port</td><td>{{.Port}}</td></tr>
      <tr><td>DNS Type</td><td>{{.Dnstype}}</td></tr>
      <tr><td>Non Standard IP</td><td>{{.Nonstdip}}</td></tr>
      <tr><td>Last Connect</td><td>{{.Lastconnect}}<br>{{.Lastconnectago}} ago</td></tr>
      <tr><td>Last Connect Status</td><td>{{.Statusstr}}</td></tr>
      <tr><td>Last Try</td><td>{{.Lasttry}}<br>{{.Lasttryago}} ago</td></tr>
      <tr><td>Crawl Start</td><td>{{.Crawlstart}}<br>{{.Crawlstartago}} ago</td></tr>
      <tr><td>Crawl Active</td><td>{{.Crawlactive}}</td></tr>
      <tr><td>Connection Fails</td><td>{{.Connectfails}}</td></tr>
      <tr><td>Remote Version</td><td>{{.Version}}</td></tr>
      <tr><td>Remote SubVersion</td><td>{{.Strversion}}</td></tr>
      <tr><td>Remote Services</td><td>{{.Services}}</td></tr>
      <tr><td>Remote Last Block</td><td>{{.Lastblock}}</td></tr>
    </table>
    </center>
    `
	// read the seeder name
	n := r.FormValue("s")
	s := getSeederByName(n)
	if s == nil {
		writeHeader(w, r)
		fmt.Fprintf(w, "No seeder found called %s", html.EscapeString(n))
		writeFooter(w, r, st)
		return
	}

	s.mtx.RLock()
	defer s.mtx.RUnlock()

	k := r.FormValue("nd")
	writeHeader(w, r)
	if _, ok := s.theList[k]; ok == false {
		fmt.Fprintf(w, "Sorry there is no Node with those details\n")
	} else {

		nd := s.theList[k]
		wt := webtemplate{
			IP:             nd.na.IP.String(),
			Port:           nd.na.Port,
			Dnstype:        nd.dns2str(),
			Nonstdip:       nd.nonstdIP.String(),
			Statusstr:      nd.statusStr,
			Lastconnect:    nd.lastConnect.String(),
			Lastconnectago: time.Since(nd.lastConnect).String(),
			Lasttry:        nd.lastTry.String(),
			Lasttryago:     time.Since(nd.lastTry).String(),
			Crawlstart:     nd.crawlStart.String(),
			Crawlstartago:  time.Since(nd.crawlStart).String(),
			Connectfails:   nd.connectFails,
			Crawlactive:    nd.crawlActive,
			Version:        nd.version,
			Strversion:     nd.strVersion,
			Services:       nd.services.String(),
			Lastblock:      nd.lastBlock,
		}

		// display details for the Node
		t := template.New("Node template")
		t, err := t.Parse(ndt)
		if err != nil {
			log.Printf("error parsing Node template %v\n", err)
		}
		err = t.Execute(w, wt)
		if err != nil {
			log.Printf("error executing Node template %v\n", err)
		}

	}
	writeFooter(w, r, st)
}

// summaryHandler displays details about one node
func summaryHandler(w http.ResponseWriter, r *http.Request) {

	st := time.Now()

	var hc struct {
		Name     string
		RG       uint32
		RGS      uint32
		CG       uint32
		CGS      uint32
		WG       uint32
		WGS      uint32
		NG       uint32
		NGS      uint32
		Total    uint32
		V4Std    uint32
		V4Non    uint32
		V6Std    uint32
		V6Non    uint32
		DNSTotal uint32
	}

	writeHeader(w, r)
	// loop through each of the seeder name from a slice so they are always returned in
	// the same order then get a pointer to the seeder struct
	for _, n := range config.order {
		s := config.seeders[n]

		hc.Name = s.name
		// fill the structs so they can be displayed via the template
		s.counts.mtx.RLock()
		hc.RG = s.counts.NdStatus[statusRG]
		hc.RGS = s.counts.NdStarts[statusRG]
		hc.CG = s.counts.NdStatus[statusCG]
		hc.CGS = s.counts.NdStarts[statusCG]
		hc.WG = s.counts.NdStatus[statusWG]
		hc.WGS = s.counts.NdStarts[statusWG]
		hc.NG = s.counts.NdStatus[statusNG]
		hc.NGS = s.counts.NdStarts[statusNG]
		hc.Total = hc.RG + hc.CG + hc.WG + hc.NG

		hc.V4Std = s.counts.DNSCounts[dnsV4Std]
		hc.V4Non = s.counts.DNSCounts[dnsV4Non]
		hc.V6Std = s.counts.DNSCounts[dnsV6Std]
		hc.V6Non = s.counts.DNSCounts[dnsV6Non]
		hc.DNSTotal = hc.V4Std + hc.V4Non + hc.V6Std + hc.V6Non
		s.counts.mtx.RUnlock()

		// we are using basic and simple html here. No fancy graphics or css
		sp := `
    <b>Stats for seeder: {{.Name}}</b>
    <center>
    <table><tr><td>
    Node Stats (count/started)<br>
    <table border=1><tr>
	<td><a href="/statusRG?s={{.Name}}">RG: {{.RG}}/{{.RGS}}</a></td>
    <td><a href="/statusCG?s={{.Name}}">CG: {{.CG}}/{{.CGS}}</a></td>
    <td><a href="/statusWG?s={{.Name}}">WG: {{.WG}}/{{.WGS}}</a></td>
    <td><a href="/statusNG?s={{.Name}}">NG: {{.NG}}/{{.NGS}}</a></td>
    <td>Total: {{.Total}}</td>
    <td><a title="Export in format consumed by Bitcoin Core contrib/seeds" href="/seeds.txt?s={{.Name}}">seeds.txt</a></td>
    </tr></table>
    </td><td>
    DNS Requests<br>
    <table border=1><tr>
	<td>V4 Std: {{.V4Std}}</td>
    <td>V4 Non: {{.V4Non}}</td>
    <td>V6 Std: {{.V6Std}}</td>
    <td>V6 Non: {{.V6Non}}</td>
    <td><a href="/dns?s={{.Name}}">Total: {{.DNSTotal}}</a></td>
    </tr></table>
    </td></tr></table>
	</center>
	`
		t := template.New("Header template")
		t, err := t.Parse(sp)
		if err != nil {
			log.Printf("error parsing summary template %v\n", err)
		}

		err = t.Execute(w, hc)
		if err != nil {
			log.Printf("error executing summary template %v\n", err)
		}
	}
	writeFooter(w, r, st)
}

// txtHandler outputs the node list in the format expected by Bitcoin Core's
// contrib/seeds script.
func txtHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain")

	// read the seeder name
	n := r.FormValue("s")
	s := getSeederByName(n)
	if s == nil {
		fmt.Fprintf(w, "No seeder found called %s\n", n)
		return
	}

	fmt.Fprintf(w, "# address                                        good  lastSuccess    %%(2h)   %%(8h)   %%(1d)   %%(7d)  %%(30d)  blocks      svcs  version\n")

	// gather all the info before writing anything to the remote browser
	s.mtx.RLock()
	defer s.mtx.RUnlock()

	for k, v := range s.theList {
		address := k

		var good int
		if v.status == statusCG {
			good = 1
		} else {
			good = 0
		}

		lastSuccess := v.lastConnect

		// Alas we don't actually measure this, so fake it.
		uptime := (100.0-float32(v.rating))/2.0 + 50.0

		blocks := v.lastBlock

		services := v.services

		version := v.version

		userAgent := v.strVersion

		fmt.Fprintf(w, "%s                                  %d   %d  %.2f%% %.2f%% %.2f%% %.2f%% %.2f%%  %d  %08x  %d %q\n", address, good, lastSuccess.Unix(), uptime, uptime, uptime, uptime, uptime, blocks, int32(services), version, userAgent)
	}
}

// writeHeader will output the standard header
func writeHeader(w http.ResponseWriter, r *http.Request) {
	// we are using basic and simple html here. No fancy graphics or css
	h1 := `
    <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
    <html><head><title>dnsseeder</title></head><body>
	<center>
	<a href="/summary">Summary</a>   
`
	fmt.Fprintf(w, "%s", h1)

	// read the seeder name
	n := r.FormValue("s")
	if n != "" {
		s := getSeederByName(n)
		if s != nil {
			fmt.Fprintf(w, "<br><b>Seeder: %s</b>", html.EscapeString(s.name))
		}
	}
	fmt.Fprintf(w, "</center><hr><br>")
}

// writeFooter will output the standard footer
func writeFooter(w http.ResponseWriter, r *http.Request, st time.Time) {

	// Footer needs to be exported for template processing to work
	var Footer struct {
		Uptime  string
		Version string
		Rt      string
	}

	f := `
	<hr>
	<center>
	<b>Version:</b> {{.Version}}
	<b>Uptime:</b> {{.Uptime}}
	<b>Request Time:</b> {{.Rt}}
	</center>
	</body></html>
	`
	Footer.Uptime = time.Since(config.uptime).String()
	Footer.Version = config.version
	Footer.Rt = time.Since(st).String()

	t := template.New("Footer template")
	t, err := t.Parse(f)
	if err != nil {
		log.Printf("error parsing template %v\n", err)
	}
	err = t.Execute(w, Footer)
	if err != nil {
		log.Printf("error executing template %v\n", err)
	}

	if config.verbose {
		log.Printf("status - processed web request: %s %s\n",
			r.RemoteAddr,
			r.RequestURI)
	}
}

/*


 */
