
function {:existential true} inv(low: int, mid: int, high: int): bool;
procedure main()
{
  var low, mid, high: int;
  var b0: bool;
  assume (low*1)+(mid*0)+(high*0)==0 && (low*0)+(mid*1)+(high*0)>=1 && (low*0)+(mid*-2)+(high*1)==0;
  while ((low*0)+(mid*1)+(high*0)>0)
  invariant inv(low, mid, high);
  {
    havoc b0;
    
    if (b0) {
        low := 1*low+0*mid+0*high+1;
mid := 0*low+1*mid+0*high+-1;
high := 0*low+0*mid+1*high+-1;

    }

  }
  assert (low*1)+(mid*0)+(high*-1)==0;
}
