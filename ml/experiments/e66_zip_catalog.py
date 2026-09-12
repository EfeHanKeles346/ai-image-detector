"""Read only the SIDD ZIP directory with bounded HTTP ranges, without extracting image bytes."""
import io
import json
from pathlib import Path
import zipfile
import requests
from experiments.e65_acquisition import digest,read,write_once
from experiments.e66_acquisition import ROOT,CONTRACT,URL


class RangeReader(io.RawIOBase):
    def __init__(self,url,size,budget=4*1024**2):
        self.url=url;self.size=size;self.position=0;self.used=0;self.budget=budget
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if position<0 or position>self.size:raise ValueError('invalid archive seek')
        self.position=position;return position
    def read(self,size=-1):
        amount=self.size-self.position if size<0 else min(size,self.size-self.position)
        if amount==0:return b''
        if self.used+amount>self.budget:raise ValueError('ZIP metadata transfer budget exceeded')
        start=self.position;end=start+amount-1
        with requests.get(self.url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(10,30)) as r:
            r.raise_for_status()
            if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                raise ValueError('server did not honor exact metadata range')
            body=b''
            for chunk in r.iter_content(min(amount,65536)):
                body+=chunk
                if len(body)>amount:raise ValueError('range body too long')
            if len(body)!=amount:raise ValueError('range body truncated')
        self.position+=amount;self.used+=amount;return body


def catalog():
    c=read(CONTRACT)
    reader=RangeReader(URL,c['row']['bytes'])
    with zipfile.ZipFile(reader) as archive:
        rows=[{'filename':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,
               'crc32':f'{i.CRC:08x}','compression':i.compress_type,'directory':i.is_dir(),
               'header_offset':i.header_offset} for i in archive.infolist()]
    result={'state':'E66_SIDD_remote_ZIP_directory_only','acquisition_contract_sha256':digest(CONTRACT),
            'code_sha256':digest(__file__),'range_metadata_bytes':reader.used,'entries':rows,
            'model_scores_created':0,'image_members_extracted':0}
    write_once(ROOT/'sidd_zip_directory.json',result)
    print(json.dumps({'entries':len(rows),'metadata_bytes':reader.used,'sample':rows[:5]},indent=2))


if __name__=='__main__':catalog()
