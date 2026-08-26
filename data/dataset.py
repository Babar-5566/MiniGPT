"""Quran parsing and canonical Surah/Ayah/Juz access utilities."""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Ayah:
    surah: int
    ayah: int
    text: str

def load_quran(path='data/quran.txt'):
    records=[]; malformed=[]; seen=set(); duplicates=[]
    with open(path,encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            line=line.rstrip('\r\n')
            if not line.strip(): continue
            parts=line.split('|',2)
            if len(parts)!=3:
                malformed.append((line_no,line)); continue
            try: s,a=int(parts[0].strip()),int(parts[1].strip())
            except ValueError: malformed.append((line_no,line)); continue
            if s<1 or s>114 or a<1:
                malformed.append((line_no,line)); continue
            key=(s,a)
            if key in seen:
                duplicates.append((line_no, key)); continue
            seen.add(key)
            records.append(Ayah(s,a,parts[2]))
    records.sort(key=lambda x:(x.surah,x.ayah))
    if duplicates:
        details=', '.join(f'{s}:{a}' for _,(s,a) in duplicates[:5])
        raise ValueError(f'Duplicate Surah/Ayah records in Quran data: {details}')
    return records,malformed

def parse_juz_mappings(path='data/quran_juz_mapping.txt'):
    out=[]; pat=re.compile(r'Juz\s+(\d+):.*?(\d+)(?:st|nd|rd|th) Surah.*?Ayah\s+(\d+) to .*?(\d+)(?:st|nd|rd|th) Surah.*?Ayah\s+(\d+)',re.I)
    for line in open(path,encoding='utf-8'):
        m=pat.search(line)
        if not m: continue
        j,ss,sa,es,ea=map(int,m.groups())
        if not (1 <= j <= 30 and ss >= 1 and es >= 1 and sa >= 1 and ea >= 1):
            raise ValueError(f'Invalid Juz mapping values: {line.strip()}')
        if (ss,sa) > (es,ea):
            raise ValueError(f'Juz {j} start is after its end')
        out.append({'juz_number':j,'start_surah':ss,'start_ayah':sa,'end_surah':es,'end_ayah':ea})
    numbers=[x['juz_number'] for x in out]
    if len(out) != 30 or set(numbers) != set(range(1,31)):
        raise ValueError(f'Expected exactly 30 Juz mappings (1-30), found {len(out)}')
    if len(set(numbers)) != len(numbers):
        raise ValueError('Duplicate Juz mapping numbers found')
    return sorted(out,key=lambda x:x['juz_number'])

class QuranDataset:
    def __init__(self,quran_path='data/quran.txt',juz_path='data/quran_juz_mapping.txt'):
        self.ayahs,self.malformed=load_quran(quran_path)
        self.mappings=parse_juz_mappings(juz_path)
        self.by_key={(a.surah,a.ayah):a for a in self.ayahs}
    def get_surah(self,n): return [a for a in self.ayahs if a.surah==n]
    def get_ayah(self,s,a): return self.by_key.get((s,a))
    def get_juz(self,n):
        if not isinstance(n, int) or not 1 <= n <= 30:
            raise ValueError(f'Invalid Juz number {n!r}; expected an integer from 1 to 30')
        m=next((x for x in self.mappings if x['juz_number']==n), None)
        if m is None:
            raise ValueError(f'Juz {n} is not available')
        return [a for a in self.ayahs if (a.surah,a.ayah)>=(m['start_surah'],m['start_ayah']) and (a.surah,a.ayah)<=(m['end_surah'],m['end_ayah'])]
    def text(self): return '\n'.join(a.text for a in self.ayahs)
