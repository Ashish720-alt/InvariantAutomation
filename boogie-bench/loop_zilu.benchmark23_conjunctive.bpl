
function {:existential true} inv(i: int, j: int): bool;
procedure main()
{
  var i, j: int;
  var b0: bool;
  assume (i*1)+(j*0)==0 && (i*0)+(j*1)==0;
  while ((i*1)+(j*0)<100)
  invariant inv(i, j);
  {
    havoc b0;
    
    if (b0) {
        i := 1*i+0*j+1;
j := 0*i+1*j+2;

    }

  }
  assert (i*0)+(j*1)==200;
}
