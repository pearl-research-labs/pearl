// 入口文件：必须先装好随机源与编码 API，再加载应用本体
import 'react-native-get-random-values';

// Hermes 部分版本缺少 WHATWG TextEncoder（@noble/@scure 密码学库依赖）。
// 仅在缺失时注入最小实现；助记词、tag 字符串均走此路径。
if (typeof globalThis.TextEncoder === 'undefined') {
  class TextEncoderPolyfill {
    encode(str) {
      // 标准 UTF-8 编码（含代理对处理）
      const out = [];
      for (let i = 0; i < str.length; i++) {
        let cp = str.charCodeAt(i);
        if (cp >= 0xd800 && cp <= 0xdbff && i + 1 < str.length) {
          const lo = str.charCodeAt(i + 1);
          if (lo >= 0xdc00 && lo <= 0xdfff) {
            cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
            i++;
          }
        }
        if (cp < 0x80) out.push(cp);
        else if (cp < 0x800) {
          out.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f));
        } else if (cp < 0x10000) {
          out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
        } else {
          out.push(
            0xf0 | (cp >> 18),
            0x80 | ((cp >> 12) & 0x3f),
            0x80 | ((cp >> 6) & 0x3f),
            0x80 | (cp & 0x3f)
          );
        }
      }
      return Uint8Array.from(out);
    }
  }
  // eslint-disable-next-line no-undef
  globalThis.TextEncoder = TextEncoderPolyfill;
}

import {registerRootComponent} from 'expo';
import App from './App';

registerRootComponent(App);
