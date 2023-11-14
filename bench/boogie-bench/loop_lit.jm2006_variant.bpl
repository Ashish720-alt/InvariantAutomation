
function {:existential true} inv(i: int, j: int, x: int, y: int, z: int): bool;
procedure main()
{
  var i, j, x, y, z: int;
  var b0: bool;
  assume (i*1)+(j*0)+(x*0)+(y*0)+(z*0)>=0 && (i*1)+(j*0)+(x*0)+(y*0)+(z*0)<=1000000 && (i*0)+(j*1)+(x*0)+(y*0)+(z*0)>=0 && (i*-1)+(j*0)+(x*1)+(y*0)+(z*0)==0 && (i*0)+(j*-1)+(x*0)+(y*1)+(z*0)==0 && (i*0)+(j*0)+(x*0)+(y*0)+(z*1)==0;
  while ((i*0)+(j*0)+(x*1)+(y*0)+(z*0)>0 || (i*0)+(j*0)+(x*1)+(y*0)+(z*0)<0)
  invariant inv(i, j, x, y, z);
  {
    havoc b0;
    
    if (b0) {
        i := 1*i+0*j+0*x+0*y+0*z+0;
j := 0*i+1*j+0*x+0*y+0*z+0;
x := 0*i+0*j+1*x+0*y+0*z+-1;
y := 0*i+0*j+0*x+1*y+0*z+-2;
z := 0*i+0*j+0*x+0*y+1*z+1;

    }

  }
  assert (i*1)+(j*-1)+(x*0)+(y*0)+(z*0)<0;
}
