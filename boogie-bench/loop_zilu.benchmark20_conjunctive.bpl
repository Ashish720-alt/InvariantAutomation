
function {:existential true} inv(i: int, n: int, sum: int): bool;
procedure main()
{
  var i, n, sum: int;
  var b0: bool;
  assume (i*1)+(n*0)+(sum*0)==0 && (i*0)+(n*1)+(sum*0)>=0 && (i*0)+(n*1)+(sum*0)<=100 && (i*0)+(n*0)+(sum*1)==0;
  while ((i*1)+(n*-1)+(sum*0)<0)
  invariant inv(i, n, sum);
  {
    havoc b0;
    
    if (b0) {
        i := 1*i+0*n+0*sum+1;
n := 0*i+1*n+0*sum+0;
sum := 1*i+0*n+1*sum+0;

    }

  }
  assert (i*0)+(n*0)+(sum*1)>=0;
}
