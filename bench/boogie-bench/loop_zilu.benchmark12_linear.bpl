
function {:existential true} inv(x: int, y: int, t: int): bool;
procedure main()
{
  var x, y, t: int;
  var b: bool;
  assume (x*1)+(y*-1)+(t*0)>0 && (x*0)+(y*1)+(t*-1)==0 || (x*1)+(y*-1)+(t*0)<0 && (x*0)+(y*1)+(t*-1)==0;
  while (b)
  invariant inv(x, y, t);
  {
    havoc b;
    
    if ((x*1)+(y*0)+(t*0)>0) {
        x := 1*x+0*y+0*t+0;
y := 1*x+1*y+0*t+0;
t := 0*x+0*y+1*t+0;

    }

    if ((x*1)+(y*0)+(t*0)<=0) {
        x := 1*x+0*y+0*t+0;
y := 0*x+1*y+0*t+0;
t := 0*x+0*y+1*t+0;

    }

  }
  assert (x*0)+(y*1)+(t*-1)>=0;
}
