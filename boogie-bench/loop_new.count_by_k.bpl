
function {:existential true} inv(i: int, k: int): bool;
procedure main()
{
  var i, k: int;
  var b0: bool;
  assume (i*1)+(k*0)==0 && (i*0)+(k*1)>=0 && (i*0)+(k*1)<=10;
  while ((i*1)+(k*-1000000)<0)
  invariant inv(i, k);
  {
    havoc b0;
    
    if (b0) {
        i := 1*i+1*k+0;
k := 0*i+1*k+0;

    }

  }
  assert (i*1)+(k*-1000000)==0;
}
