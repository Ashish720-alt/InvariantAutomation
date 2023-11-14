
function {:existential true} inv(x: int, y: int, z: int): bool;
procedure main()
{
  var x, y, z: int;
  var b: bool;
  assume (x*0)+(y*1)+(z*0)>0 || (x*1)+(y*0)+(z*0)>0 || (x*0)+(y*0)+(z*1)>0;
  while (b)
  invariant inv(x, y, z);
  {
    havoc b;
    
    if ((x*1)+(y*0)+(z*0)>0 && (x*0)+(y*1)+(z*0)>0) {
        x := 1*x+0*y+0*z+1;
y := 0*x+1*y+0*z+1;
z := 0*x+0*y+1*z+0;

    }

    if ((x*1)+(y*0)+(z*0)>0 && (x*0)+(y*1)+(z*0)<=0) {
        x := 1*x+0*y+0*z+1;
y := 0*x+1*y+0*z+0;
z := 0*x+0*y+1*z+1;

    }

    if ((x*1)+(y*0)+(z*0)<=0 && (x*0)+(y*1)+(z*0)>0) {
        x := 1*x+0*y+0*z+0;
y := 0*x+1*y+0*z+1;
z := 0*x+0*y+1*z+0;

    }

    if ((x*1)+(y*0)+(z*0)<=0 && (x*0)+(y*1)+(z*0)<=0) {
        x := 1*x+0*y+0*z+0;
y := 0*x+1*y+0*z+0;
z := 0*x+0*y+1*z+1;

    }

  }
  assert (x*0)+(y*1)+(z*0)>0;
}
