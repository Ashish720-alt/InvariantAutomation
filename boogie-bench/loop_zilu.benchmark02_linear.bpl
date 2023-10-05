
function {:existential true} inv(n: int, i: int, l: int): bool;
procedure main()
{
  var n, i, l: int;
  var b0: bool;
  assume (n*0)+(i*0)+(l*1)>0 && (n*0)+(i*1)+(l*-1)==0;
  while ((n*-1)+(i*1)+(l*0)<0)
  invariant inv(n, i, l);
  {
    havoc b0;
    
    if (b0) {
        n := 1*n+0*i+0*l+0;
i := 0*n+1*i+0*l+1;
l := 0*n+0*i+1*l+0;

    }

  }
  assert (n*0)+(i*0)+(l*1)>=1;
}
