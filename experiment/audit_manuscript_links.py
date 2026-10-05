"""Read active TeX inputs recursively; audit definitions, calls, and compiled labels."""
from pathlib import Path
from collections import Counter
import json
import re
import argparse

ROOT=Path(__file__).resolve().parent.parent
REF=re.compile(r'\\(?:[cC]ref|[cC]refrange|ref|eqref|autoref|pageref)\*?\s*\{([^}]+)\}')

def active_sources():
    visited=set(); ordered=[]
    def visit(path):
        path=path.resolve()
        if path in visited: return
        visited.add(path); ordered.append(path)
        for m in re.finditer(r'\\(?:input|include)\s*\{([^}]+)\}',path.read_text(encoding='utf-8')):
            child=path.parent/m.group(1)
            if not child.suffix: child=child.with_suffix('.tex')
            assert child.is_file(),str(child)
            visit(child)
    visit(ROOT/'main.tex'); return ordered

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='working/cross_reference_audit.json')
    parser.add_argument('--strict',action='store_true'); args=parser.parse_args()
    labels={}; references=[]; figures=[]; hardcoded=[]
    for path in active_sources():
        for n,raw in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            line=re.split(r'(?<!\\)%',raw)[0]; location=dict(file=str(path.relative_to(ROOT)),line=n)
            for m in re.finditer(r'\\label\s*\{([^}]+)\}',line):
                labels.setdefault(m.group(1),[]).append(location)
            for m in REF.finditer(line):
                for key in m.group(1).split(','):
                    references.append(dict(label=key.strip(),**location))
            for m in re.finditer(r'\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}',line):
                name=m.group(1); candidates=[ROOT/name,ROOT/'figures'/name]
                found=next((p for p in candidates if p.is_file()),None)
                figures.append(dict(name=name,exists=found is not None,**location))
            if re.search(r'\b(?:Figure|Table|Fig\.|Tab\.|Algorithm|Proposition|Definition)s?\s*(?:~|\s)\s*\d',line):
                hardcoded.append(dict(text=line.strip(),**location))
    calls=Counter(r['label'] for r in references)
    aux=(ROOT/'main.aux').read_text(encoding='utf-8',errors='replace') if (ROOT/'main.aux').exists() else ''
    compiled={m.group(1):dict(number=m.group(2),page=int(m.group(3)))
              for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}',aux)}
    inventory=[dict(label=key,source=locs[0],calls=calls[key],compiled=compiled.get(key))
               for key,locs in labels.items() if key.startswith(('fig:','tab:','alg:'))]
    result=dict(sources=[str(p.relative_to(ROOT)) for p in active_sources()],
        definitions=len(labels),reference_calls=len(references),inventory=inventory,
        duplicate_labels={k:v for k,v in labels.items() if len(v)>1},
        undefined_references=[r for r in references if r['label'] not in labels],
        unreferenced_objects=[r for r in inventory if not r['calls']],
        uncompiled_objects=[r for r in inventory if r['compiled'] is None],
        hardcoded_number_references=hardcoded,missing_graphics=[f for f in figures if not f['exists']],
        graphics=figures)
    output=ROOT/args.output; output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inventory','graphics']},indent=2))
    if args.strict:
        assert not any(result[k] for k in ['duplicate_labels','undefined_references','unreferenced_objects',
            'uncompiled_objects','hardcoded_number_references','missing_graphics'])

if __name__=='__main__': main()
