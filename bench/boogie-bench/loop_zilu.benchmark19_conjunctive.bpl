
function {:existential true} inv(j: int, k: int, n: int): bool;
procedure main()
{
  var j, k, n: int;
  var b0: bool;
  assume (j*1)+(k*0)+(n*-1)==0 && (j*0)+(k*1)+(n*-1)==0 && (j*0)+(k*0)+(n*1)>0;
  while ((j*0)+(k*0)+(n*1)>0 && (j*1)+(k*0)+(n*0)>0)
  invariant inv(j, k, n);
  {
    havoc b0;
    
    if (b0) {
        j := 1*j+0*k+0*n+-1;
k := 0*j+1*k+0*n+-1;
n := 0*j+0*k+1*n+0;

    }

  }
  assert (j*0)+(k*1)+(n*0)==0 && (j*0)+(k*1)+(n*0)==0;
}
