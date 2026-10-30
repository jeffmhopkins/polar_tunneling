"""Fill tpl.html with data.json (written by build.py) to make the publishable page."""
import sys
tpl, data, out = sys.argv[1:4]
open(out, "w").write(open(tpl).read().replace("__DATA__", open(data).read().strip(), 1))
