
function {:existential true} inv(i: int, j: int, x: int, y: int): bool;
procedure main()
{
  var i, j, x, y: int;
  var b0: bool;
  assume (i*1)+(j*0)+(x*0)+(y*0)>=0 && (i*0)+(j*1)+(x*0)+(y*0)>=0 && (i*-1)+(j*0)+(x*1)+(y*0)==0 && (i*0)+(j*-1)+(x*0)+(y*1)==0;
  while ((i*0)+(j*0)+(x*1)+(y*0)>0 || (i*0)+(j*0)+(x*1)+(y*0)<0)
  invariant inv(i, j, x, y);
  {
    havoc b0;
    
    if (b0) {
        i := 1*i+0*j+0*x+0*y+0;
j := 0*i+1*j+0*x+0*y+0;
x := 0*i+0*j+1*x+0*y+-1;
y := 0*i+0*j+0*x+1*y+-1;

    }

  }
  assert (i*1)+(j*-1)+(x*0)+(y*0)<0;
}
