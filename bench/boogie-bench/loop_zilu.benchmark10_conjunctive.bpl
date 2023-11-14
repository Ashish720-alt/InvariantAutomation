
function {:existential true} inv(i: int, c: int): bool;
procedure main()
{
  var i, c: int;
  var b0: bool;
  assume (i*0)+(c*1)==0 && (i*1)+(c*0)==0;
  while ((i*1)+(c*0)<100 && (i*1)+(c*0)>-1)
  invariant inv(i, c);
  {
    havoc b0;
    
    if (b0) {
        i := 1*i+0*c+1;
c := 1*i+1*c+0;

    }

  }
  assert (i*0)+(c*1)>=0;
}
