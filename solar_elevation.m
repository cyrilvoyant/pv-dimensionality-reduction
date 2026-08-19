function el = solar_elevation(dt, lat, lon)
% Elevation du soleil (en degres) pour des instants dt donnes en UTC, a la
% latitude lat et la longitude lon (degres, Est positif). Formule simplifiee
% (equation du temps negligee) : largement suffisante pour distinguer jour et
% nuit a 30 min de resolution. el <= 0 => soleil sous l'horizon => nuit.
    N      = day(dt, 'dayofyear');
    hUTC   = hour(dt) + minute(dt)/60 + second(dt)/3600;
    decl   = 23.45 * sind(360 * (284 + N) / 365);      % declinaison solaire
    tsolar = hUTC + lon/15;                             % temps solaire local (approx)
    H      = 15 * (tsolar - 12);                        % angle horaire (degres)
    el     = asind( sind(lat).*sind(decl) + cosd(lat).*cosd(decl).*cosd(H) );
end
