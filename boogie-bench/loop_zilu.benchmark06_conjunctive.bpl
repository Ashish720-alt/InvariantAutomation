
function {:existential true} inv(i: int, j: int, x: int, y: int, k: int): bool;
procedure main()
{
  var i, j, x, y, k: int;
  var b: bool;
  assume (i*0)+(j*0)+(x*1)+(y*1)+(k*-1)==0 && (i*0)+(j*1)+(x*0)+(y*0)+(k*0)==0;
  while (b)
  invariant inv(i, j, x, y, k);
  {
    havoc b;
    
    if ((i*-1)+(j*1)+(x*0)+(y*0)+(k*0)==0) {
        i := 1*i+0*j+0*x+0*y+0*k+0;
j := 0*i+1*j+0*x+0*y+0*k+1;
x := 0*i+0*j+1*x+0*y+0*k+1;
y := 0*i+0*j+0*x+1*y+0*k+-1;
k := 0*i+0*j+0*x+0*y+1*k+0;

    }

    if ((i*-1)+(j*1)+(x*0)+(y*0)+(k*0)<0 || (i*1)+(j*-1)+(x*0)+(y*0)+(k*0)<0) {
        i := 1*i+0*j+0*x+0*y+0*k+0;
j := 0*i+1*j+0*x+0*y+0*k+1;
x := 0*i+0*j+1*x+0*y+0*k+-1;
y := 0*i+0*j+0*x+1*y+0*k+1;
k := 0*i+0*j+0*x+0*y+1*k+0;

    }

  }
  assert (i*0)+(j*0)+(x*1)+(y*1)+(k*-1)==0;
}
