'use client';

import { useEffect, useRef } from 'react';
import { useTheme } from '@/components/ThemeProvider';
import travels from '@/data/travel.json';

type Airport = { name: string; iata: string; icao: string; coordinates: [number, number] };
type Route   = { from: Airport; to: Airport };

const flightData = travels as Route[];

const CITY_NAMES: Record<string, string> = {
  PVG: 'Shanghai', HKG: 'Hong Kong', DOH: 'Doha', ATL: 'Atlanta',
  CUN: 'Cancún', MEX: 'Mexico City', NLU: 'Mexico City', IAH: 'Houston',
  BWI: 'Washington DC', KEF: 'Reykjavik', PKX: 'Beijing', PEK: 'Beijing',
  KMG: 'Kunming', SZX: 'Shenzhen', CAN: 'Guangzhou', XIY: "Xi'an",
  IST: 'Istanbul', CLT: 'Charlotte', SFO: 'San Francisco', FLL: 'Fort Lauderdale',
  LIM: 'Lima', CUZ: 'Cusco', MDW: 'Chicago', CLE: 'Cleveland',
  SAN: 'San Diego', LAX: 'Los Angeles', SEA: 'Seattle', HNL: 'Honolulu',
  NRT: 'Tokyo', KIX: 'Osaka', ICN: 'Seoul', LGA: 'New York',
  BKK: 'Bangkok', CXR: 'Nha Trang', AUS: 'Austin', DEN: 'Denver', PHX: 'Phoenix',
  YYZ: 'Toronto', DFW: 'Dallas', JFK: 'New York (JFK)', FCO: 'Rome',
  BCN: 'Barcelona', CDG: 'Paris',
};

const airportMap = new Map<string, Airport>();
flightData.forEach((f) => {
  airportMap.set(f.from.iata, f.from);
  airportMap.set(f.to.iata, f.to);
});
const cityMap = new Map<string, Airport>();
airportMap.forEach((apt) => {
  const city = CITY_NAMES[apt.iata] ?? apt.name;
  if (!cityMap.has(city)) cityMap.set(city, apt);
});

const routes: Array<{ from: Airport; to: Airport }> = [];
const _routeSet = new Set<string>();
flightData.forEach((f) => {
  const key = [f.from.iata, f.to.iata].sort().join('-');
  if (!_routeSet.has(key)) {
    _routeSet.add(key);
    routes.push(f);
  }
});

export default function TravelMap() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const mapRef = useRef<HTMLDivElement>(null);
  const instanceRef = useRef<any>(null);

  useEffect(() => {
    if (!mapRef.current || instanceRef.current) return;

    import('leaflet').then((L) => {
      if (!mapRef.current || instanceRef.current) return;

      const map = L.map(mapRef.current!, {
        center: [20, 10],
        zoom: 2,
        minZoom: 2,
        maxZoom: 10,
        zoomControl: true,
        worldCopyJump: true,
      });
      instanceRef.current = map;

      const tileUrl = isDark
        ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
        : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';

      L.tileLayer(tileUrl, {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19,
      }).addTo(map);

      const arcColor = isDark ? '#a0b4d0' : '#334155';
      const dotColor = isDark ? '#e2e8f0' : '#1e293b';

      // Draw arcs as quadratic bezier polylines, splitting at the antimeridian
      routes.forEach(({ from, to }) => {
        const [lng1, lat1] = from.coordinates;
        const [lng2Raw, lat2] = to.coordinates;

        // Adjust lng2 so we always take the short path (may go outside [-180,180])
        let lng2 = lng2Raw;
        if (lng2 - lng1 > 180) lng2 -= 360;
        if (lng2 - lng1 < -180) lng2 += 360;

        const dlat = lat2 - lat1;
        const dlng = lng2 - lng1;
        const midLat = (lat1 + lat2) / 2 - dlng * 0.15;
        const midLng = (lng1 + lng2) / 2 + dlat * 0.15;

        // Sample bezier in continuous lng space, then wrap each point to [-180,180]
        // Split into a new segment wherever consecutive wrapped lngs jump > 180 (antimeridian)
        const segments: [number, number][][] = [[]];
        for (let i = 0; i <= 64; i++) {
          const t = i / 64;
          const lat = (1 - t) * (1 - t) * lat1 + 2 * (1 - t) * t * midLat + t * t * lat2;
          const rawLng = (1 - t) * (1 - t) * lng1 + 2 * (1 - t) * t * midLng + t * t * lng2;
          const lng = ((rawLng + 180) % 360 + 360) % 360 - 180; // wrap to [-180,180]

          const cur = segments[segments.length - 1];
          if (cur.length > 0 && Math.abs(lng - cur[cur.length - 1][1]) > 180) {
            segments.push([[lat, lng]]);
          } else {
            cur.push([lat, lng]);
          }
        }

        const style = { color: arcColor, weight: 1.2, opacity: 0.55 };
        segments.forEach(seg => { if (seg.length > 1) L.polyline(seg, style).addTo(map); });
      });

      cityMap.forEach((apt, city) => {
        const [lng, lat] = apt.coordinates;
        const circle = L.circleMarker([lat, lng], {
          radius: 4,
          color: dotColor,
          fillColor: dotColor,
          fillOpacity: 1,
          weight: 1.5,
        }).addTo(map);
        circle.bindTooltip(city, {
          permanent: false,
          direction: 'top',
          className: 'leaflet-city-tooltip',
        });
      });
    });

    return () => {
      instanceRef.current?.remove();
      instanceRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const map = instanceRef.current;
    if (!map) return;
    import('leaflet').then((L) => {
      map.eachLayer((layer: any) => {
        if (layer._url) map.removeLayer(layer);
      });
      const tileUrl = isDark
        ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
        : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
      L.tileLayer(tileUrl, {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19,
      }).addTo(map);
    });
  }, [isDark]);

  return (
    <div className="w-full">
      <div
        ref={mapRef}
        style={{ width: '100%', height: '480px', borderRadius: '4px', overflow: 'hidden' }}
      />
      <style>{`
        .leaflet-city-tooltip {
          background: rgba(0,0,0,0.75);
          border: none;
          border-radius: 3px;
          color: #fff;
          font-size: 11px;
          padding: 2px 8px;
          white-space: nowrap;
          box-shadow: none;
        }
        .leaflet-city-tooltip::before { display: none; }
      `}</style>
    </div>
  );
}
