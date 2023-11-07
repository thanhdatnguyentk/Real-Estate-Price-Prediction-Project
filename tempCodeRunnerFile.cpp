   if ( x < 1 || x > 30 )
    {
        cout << "So " << x << " khoxg nam trong khoang [1,30].";
    }
    else
    {
        cout << fibo(x);
    }
}

long long fibo(long long x)
{
    if (x == 1 || x == 2)
        return 1;
    else    
        return fibo(x-1) + fibo(x-2);
}