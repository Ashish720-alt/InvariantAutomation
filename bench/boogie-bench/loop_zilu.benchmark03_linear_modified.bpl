
function {:existential true} inv(x: int, y: int, i: int, j: int): bool;
procedure main()
{
  var x, y, i, j: int;
  var b: bool;var b0: bool;
  assume (x*1)+(y*0)+(i*0)+(j*0)==0 && (x*0)+(y*1)+(i*0)+(j*0)==0 && (x*0)+(y*0)+(i*1)+(j*0)==0 && (x*0)+(y*0)+(i*0)+(j*1)==0;
  while (b)
  invariant inv(x, y, i, j);
  {
    havoc b;havoc b0;
    
    if (b0) {
        x := 1*x+0*y+0*i+0*j+1;
y := 0*x+1*y+0*i+0*j+1;
i := 1*x+0*y+1*i+0*j+0;
j := 0*x+1*y+0*i+1*j+0;

    }

  }
  assert (x*0)+(y*0)+(i*-1)+(j*1)>=0;
}
