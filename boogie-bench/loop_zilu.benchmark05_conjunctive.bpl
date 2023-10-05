
function {:existential true} inv(x: int, y: int, n: int): bool;
procedure main()
{
  var x, y, n: int;
  
  assume (x*1)+(y*0)+(n*0)>=0 && (x*1)+(y*-1)+(n*0)<=0 && (x*0)+(y*1)+(n*-1)<0;
  while ((x*1)+(y*0)+(n*-1)<0)
  invariant inv(x, y, n);
  {
    
    
    if ((x*1)+(y*-1)+(n*0)<0) {
        x := 1*x+0*y+0*n+1;
y := 0*x+1*y+0*n+0;
n := 0*x+0*y+1*n+0;

    }

    if ((x*1)+(y*-1)+(n*0)>=0) {
        x := 1*x+0*y+0*n+1;
y := 0*x+1*y+0*n+1;
n := 0*x+0*y+1*n+0;

    }

  }
  assert (x*0)+(y*1)+(n*-1)==0;
}
