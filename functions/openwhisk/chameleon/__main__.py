from time import time

import six
from chameleon import PageTemplate

BIGTABLE_ZPT = """\
<table xmlns="http://www.w3.org/1999/xhtml"
xmlns:tal="http://xml.zope.org/namespaces/tal">
<tr tal:repeat="row python: options['table']">
<td tal:repeat="c python: row.values()">
<span tal:define="d python: c + 1"
tal:attributes="class python: 'column-' + %s(d)"
tal:content="python: d" />
</td>
</tr>
</table>""" % six.text_type.__name__

cold = True


def main(args):
    global cold
    was_cold = cold
    cold = False
    num_of_rows = int(args.get("num_of_rows", 10))
    num_of_cols = int(args.get("num_of_cols", 10))

    start = time()
    tmpl = PageTemplate(BIGTABLE_ZPT)

    data = {}
    for i in range(num_of_cols):
        data[str(i)] = i

    table = [data for x in range(num_of_rows)]
    options = {'table': table}

    data = tmpl.render(options=options)
    latency = time() - start

    return {"body": {'latency': latency, 'data': data, "cold": was_cold}}

if __name__ == '__main__':
    main({})
