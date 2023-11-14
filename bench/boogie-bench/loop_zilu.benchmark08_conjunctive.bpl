
function {:existential true} inv(n: int, sum: int, i: int): bool;
procedure main()
{
  var n, sum, i: int;
  var b0: bool;
  assume (n*1)+(sum*0)+(i*0)>=0 && (n*0)+(sum*1)+(i*0)==0 && (n*0)+(sum*0)+(i*1)==0;
  while ((n*-1)+(sum*0)+(i*1)<0)
  invariant inv(n, sum, i);
  {
    havoc b0;
    
    if (b0) {
        n := 1*n+0*sum+0*i+0;
sum := 0*n+1*sum+1*i+0;
i := 0*n+0*sum+1*i+1;

    }

  }
  assert (n*0)+(sum*1)+(i*0)>=0;
}
