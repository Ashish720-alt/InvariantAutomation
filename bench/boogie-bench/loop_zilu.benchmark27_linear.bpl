
function {:existential true} inv(i: int, j: int, k: int): bool;
procedure main()
{
  var i, j, k: int;
  var b0: bool;
  assume (i*1)+(j*-1)+(k*0)<0 && (i*-1)+(j*1)+(k*1)>0;
  while ((i*1)+(j*-1)+(k*0)<0)
  invariant inv(i, j, k);
  {
    havoc b0;
    
    if (b0) {
        i := 1*i+0*j+0*k+1;
j := 0*i+1*j+0*k+0;
k := 0*i+0*j+1*k+1;

    }

  }
  assert (i*0)+(j*0)+(k*1)>0;
}
