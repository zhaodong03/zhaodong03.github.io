import type { Metadata } from 'next';
import Image from 'next/image';
import TravelMap from '@/components/TravelMap';

export const metadata: Metadata = {
  title: 'Beyond Work',
  description: 'Life outside of work — hobbies, interests, and travels.',
};

const hobbies = [
  {
    name: 'Alpine Skiing',
    emoji: '⛷️',
    description: 'Skiing the PNW — Mt. Hood and Mission Ridge.',
    photo: '/assets/img/mthood_ski.jpg',
    photoAlt: 'Skiing at Mt. Hood',
  },
  {
    name: 'Archery',
    emoji: '🏹',
    description: '3D tournament and target shooting with a compound bow.',
    photo: null,
    photoAlt: null,
  },
  {
    name: 'Fishing',
    emoji: '🎣',
    description: 'Offshore fishing in the Gulf of Mexico and the Pacific Ocean.',
    photo: null,
    photoAlt: null,
  },
];

const travelDestinations = [
  { country: 'Iceland', emoji: '🇮🇸', cover: '/assets/img/travel/Iceland/01.jpg', note: 'Photo: danni_arndt photography' },
  { country: 'Japan', emoji: '🇯🇵', cover: '/assets/img/travel/Japan/01.jpg', note: null },
  { country: 'Turkey', emoji: '🇹🇷', cover: '/assets/img/travel/Turkey/01.jpg', note: null },
  { country: 'Mexico', emoji: '🇲🇽', cover: '/assets/img/travel/Mexico/01.jpg', note: null },
  { country: 'Peru', emoji: '🇵🇪', cover: '/assets/img/travel/Peru/01.jpg', note: null },
  { country: 'China', emoji: '🇨🇳', cover: '/assets/img/travel/China/01.jpeg', note: null },
  { country: 'United States', emoji: '🇺🇸', cover: '/assets/img/travel/US/01.jpg', note: null },
];

export default function BeyondWorkPage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12 animate-fade-in">

      {/* Page header */}
      <div className="border-b border-stone-200 pb-8 mb-12">
        <p className="section-heading">Outside the Office</p>
        <h1 className="font-serif text-4xl md:text-5xl font-black text-stone-900 leading-none">
          Beyond Work
        </h1>
      </div>

      {/* Hobbies */}
      <section className="mb-16">
        <p className="section-heading">Interests</p>
        <div className="divide-y divide-stone-200 border-t border-b border-stone-200">
          {hobbies.map(({ name, emoji, description, photo, photoAlt }) => (
            <div key={name} className="py-8 flex flex-col md:flex-row gap-8 items-start">
              <div className="flex-1">
                <h2 className="font-serif text-2xl font-bold text-stone-900 mb-2">
                  {emoji} {name}
                </h2>
                <p className="text-stone-600 text-sm leading-relaxed">{description}</p>
              </div>
              {photo && (
                <div className="relative w-full md:w-56 shrink-0 overflow-hidden border border-stone-200" style={{ aspectRatio: '3/4' }}>
                  <Image
                    src={photo}
                    alt={photoAlt ?? name}
                    fill
                    className="object-cover object-top"
                    sizes="(max-width: 768px) 100vw, 224px"
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Travel */}
      <section>
        <p className="section-heading">Travels</p>
        <div className="mb-8"><TravelMap /></div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-px bg-stone-200">
          {travelDestinations.map(({ country, emoji, cover, note }) => (
            <div key={country} className="group relative aspect-square bg-stone-100 overflow-hidden">
              <Image
                src={cover}
                alt={`Travel photo from ${country}`}
                fill
                className="object-cover transition-all duration-500 group-hover:scale-105"
                sizes="(max-width: 640px) 50vw, (max-width: 768px) 33vw, 25vw"
              />
              <div className="absolute inset-0 bg-stone-900/0 group-hover:bg-stone-900/40 transition-all duration-300" />
              <div className="absolute bottom-0 left-0 right-0 p-3 translate-y-full group-hover:translate-y-0 transition-transform duration-300">
                <p className="text-white text-sm font-semibold leading-tight drop-shadow">
                  {emoji} {country}
                </p>
                {note && (
                  <p className="text-white/70 text-xs mt-0.5 drop-shadow">{note}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
