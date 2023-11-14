
function {:existential true} inv(i: int, j: int): bool;
procedure main()
{
  var i, j: int;
  
  assume (i*1)+(j*-1)<0 && (i*1)+(j*1)>0 && (i*0)+(j*1)>=0 || (i*1)+(j*-1)>0 && (i*1)+(j*1)<0 && (i*0)+(j*1)>=0;
  while ((i*1)+(j*-1)<0)
  invariant inv(i, j);
  {
    
    
    if ((i*-2)+(j*1)<0) {
        i := -1*i+1*j+0;
j := 1*i+0*j+0;

    }

    if ((i*-2)+(j*1)>=0) {
        i := 1*i+0*j+0;
j := -1*i+1*j+0;

    }

  }
  assert (i*-1)+(j*1)==0;
}
