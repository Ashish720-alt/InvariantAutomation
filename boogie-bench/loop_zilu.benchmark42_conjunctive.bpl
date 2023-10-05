
function {:existential true} inv(x: int, y: int, z: int): bool;
procedure main()
{
  var x, y, z: int;
  var b0: bool;
  assume (x*1)+(y*-1)+(z*0)==0 && (x*1)+(y*0)+(z*0)>=0 && (x*1)+(y*1)+(z*1)==0;
  while ((x*1)+(y*0)+(z*0)>0)
  invariant inv(x, y, z);
  {
    havoc b0;
    
    if (b0) {
        x := 1*x+0*y+0*z+-1;
y := 0*x+1*y+0*z+-1;
z := 0*x+0*y+1*z+2;

    }

  }
  assert (x*0)+(y*0)+(z*1)<=0;
}
