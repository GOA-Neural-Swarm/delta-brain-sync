const crypto = require('crypto');
class HDC {
    constructor(d=10000){this.d=d}
    gen(text){
        let v=new Uint8Array(this.d);
        let h=crypto.createHash('sha256').update(text).digest();
        for(let i=0;i<this.d;i++) v[i]=h[i%h.length]%2;
        return v;
    }
}
module.exports = new HDC();
