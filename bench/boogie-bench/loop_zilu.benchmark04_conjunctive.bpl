
function {:existential true} inv(k: int, j: int, n: int): bool;
procedure main()
{
  var k, j, n: int;
  var b0: bool;
  assume (k*0)+(j*0)+(n*1)>=1 && (k*1)+(j*0)+(n*-1)>=0 && (k*0)+(j*1)+(n*0)==0;
  while ((k*0)+(j*1)+(n*-1)<=-1)
  invariant inv(k, j, n);
  {
    havoc b0;
    
    if (b0) {
        k := 1*k+0*j+0*n+-1;
j := 0*k+1*j+0*n+1;
n := 0*k+0*j+1*n+0;

    }

  }
  assert (k*1)+(j*0)+(n*0)>=0;
}
