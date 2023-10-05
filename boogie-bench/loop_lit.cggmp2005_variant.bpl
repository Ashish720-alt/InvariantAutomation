
function {:existential true} inv(lo: int, mid: int, hi: int): bool;
procedure main()
{
  var lo, mid, hi: int;
  var b0: bool;
  assume (lo*1)+(mid*0)+(hi*0)==0 && (lo*0)+(mid*1)+(hi*0)>0 && (lo*0)+(mid*1)+(hi*0)<1000000 && (lo*0)+(mid*-2)+(hi*1)==0;
  while ((lo*0)+(mid*1)+(hi*0)>0)
  invariant inv(lo, mid, hi);
  {
    havoc b0;
    
    if (b0) {
        lo := 1*lo+0*mid+0*hi+1;
mid := 0*lo+1*mid+0*hi+-1;
hi := 0*lo+0*mid+1*hi+-1;

    }

  }
  assert (lo*1)+(mid*0)+(hi*-1)==0;
}
