// ISO 3166-1 numeric codes for visited countries (matches world-atlas topojson)
export const visitedCountryIds = new Set([
  '352', // Iceland
  '392', // Japan
  '792', // Turkey
  '484', // Mexico
  '604', // Peru
  '156', // China
  '840', // USA
  '634', // Qatar
  '410', // South Korea
  '344', // Hong Kong
]);

export interface City {
  name: string;
  country: string;
  coordinates: [number, number]; // [lng, lat]
}

export const visitedCities: City[] = [
  { name: 'Reykjavik', country: 'Iceland', coordinates: [-21.9426, 64.1355] },
  { name: 'Tokyo', country: 'Japan', coordinates: [139.6917, 35.6895] },
  { name: 'Istanbul', country: 'Turkey', coordinates: [28.9784, 41.0082] },
  { name: 'Mexico City', country: 'Mexico', coordinates: [-99.1332, 19.4326] },
  { name: 'Lima', country: 'Peru', coordinates: [-77.0428, -12.0464] },
  { name: 'Beijing', country: 'China', coordinates: [116.4074, 39.9042] },
  { name: 'Atlanta', country: 'USA', coordinates: [-84.388, 33.749] },
  { name: 'Austin', country: 'USA', coordinates: [-97.7431, 30.2672] },
];
