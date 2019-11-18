function [F] = cutestUserfun (x)

[F] = cutest_cons(x);
[obj] = cutest_obj(x);

F = [ F; obj ];

