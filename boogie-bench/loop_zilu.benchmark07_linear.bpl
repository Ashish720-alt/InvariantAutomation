function {:existential true} inv(i: int, n: int, k: int): bool;

procedure main()
{
  var n, k, i: int;
  var flag: bool;
  havoc flag;
  i := 0;
  assume n>0 && n<10;

  while (i<n)
  invariant inv(i,n,k);
  {
    i := i+1;
    if (flag) {
        k := k + 4000;
    } else {
        k := k + 2000;
    }
  }
  assert k > n; 
}
