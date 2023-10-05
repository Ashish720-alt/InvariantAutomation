
function {:existential true} inv(i: int, k: int, n: int): bool;
procedure main()
{
  var i, k, n: int;
  var b0: bool;
  assume (i*1)+(k*0)+(n*0)==0 && (i*0)+(k*1)+(n*0)==0;
  while ((i*1)+(k*0)+(n*-1)<0)
  invariant inv(i, k, n);
  {
    havoc b0;
    
    if (b0) {
        i := 1*i+0*k+0*n+1;
k := 0*i+1*k+0*n+1;
n := 0*i+0*k+1*n+0;

    }

  }
  assert (i*0)+(k*1)+(n*-1)>=0;
}
