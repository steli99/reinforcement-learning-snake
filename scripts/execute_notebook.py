"""Execute ordinary Python notebook cells sequentially without a Jupyter server.

Usage: python scripts/execute_notebook.py main.ipynb
Outputs (including figures) are embedded in the notebook. No external dataset is required.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import ast,base64,contextlib,io,json,sys,time,traceback
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

path=Path(sys.argv[1]).resolve()
os.chdir(path.parent)
sys.path.insert(0,str(path.parent))
nb=nbformat.read(path,as_version=4)
namespace={'__name__':'__main__'}
start=time.time(); count=0; status='passed'; error=''
outputs=[]
def figures(*args,**kwargs):
    for num in plt.get_fignums():
        buffer=io.BytesIO()
        plt.figure(num).savefig(buffer,format='png',dpi=90,bbox_inches='tight')
        outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode()}))
        plt.close(num)
plt.show=figures
for i,cell in enumerate(nb.cells):
    if cell.cell_type!='code':continue
    print(path.stem,'cell',i,flush=True)
    count+=1;cell.execution_count=count;outputs=[];stream=io.StringIO()
    try:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            tree=ast.parse(cell.source)
            final=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
            exec(compile(tree,f'{path.name}:cell_{i}','exec'),namespace)
            if final is not None:
                value=eval(compile(ast.Expression(final.value),f'{path.name}:cell_{i}','eval'),namespace)
                if value is not None:
                    data={'text/plain':repr(value)}
                    if hasattr(value,'_repr_html_'):
                        html=value._repr_html_()
                        if html:data['text/html']=html
                    outputs.append(nbformat.v4.new_output('execute_result',data=data,execution_count=count))
            figures()
    except Exception as exc:
        status='failed';error=traceback.format_exc()
        outputs.append(nbformat.v4.new_output('error',ename=type(exc).__name__,evalue=str(exc),traceback=error.splitlines()))
    if stream.getvalue():outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
    cell.outputs=outputs
    if status=='failed':print(error,flush=True);break
nbformat.write(nb,path)
result={'notebook':path.name,'status':status,'seconds':round(time.time()-start,2),
        'execution':'Sequential Python cells, matplotlib Agg; no cells skipped', 'error':error}
report=path.parent/'docs'/f'{path.stem}_execution.json'
report.parent.mkdir(exist_ok=True)
report.write_text(json.dumps(result,indent=2))
print('FINISHED',path.name,status,result['seconds'],flush=True)
sys.exit(0 if status=='passed' else 1)
