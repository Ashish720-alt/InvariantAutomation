
function {:existential true} inv(i: int, j: int, r: int): bool;
procedure main()
{
  var i, j, r: int;
  var b0: bool;
  assume (i*-1)+(j*-1)+(r*1)>0;
  while ((i*1)+(j*0)+(r*0)>0)
  invariant inv(i, j, r);
  {
    havoc b0;
    
    if (b0) {
        i := 1*i+0*j+0*r+-1;
j := 0*i+1*j+0*r+1;
r := 0*i+0*j+1*r+0;

    }

  }
  assert (i*-1)+(j*-1)+(r*1)>0;
}
