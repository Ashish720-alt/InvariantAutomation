
function {:existential true} inv(i: int, k: int): bool;
procedure main()
{
  var i, k: int;
  var b: bool;var b0: bool;
  assume (i*0)+(k*-1)<=0 && (i*0)+(k*1)<=1 && (i*1)+(k*0)==1;
  while (b)
  invariant inv(i, k);
  {
    havoc b;havoc b0;
    
    if (b0) {
        i := 1*i+0*k+1;
k := 0*i+1*k+-1;

    }

  }
  assert (i*-1)+(k*-1)<=-1 && (i*1)+(k*1)<=2 && (i*1)+(k*0)>=1;
}
