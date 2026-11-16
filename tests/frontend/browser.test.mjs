import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const exec = promisify(execFile);
test('Chromium confirmation, identity, single-flight and unknown delivery', async () => {
  const ui = (await readFile(new URL('../../js/telegram_image_context.js', import.meta.url), 'utf8')).split('\n').slice(2).join('\n');
  const script = `
    const calls=[]; let extension, resolveSend;
    const image={filename:'test.png',subfolder:'nested',type:'output'};
    const app={nodeOutputs:{7:{images:[image]}},registerExtension:x=>extension=x};
    const api={apiURL:p=>p,fetchApi:(url,opts)=>{if(url.endsWith('/preview')) return fetch('/preview'); calls.push({url,opts});return new Promise(resolve=>resolveSend=resolve)}};
    ${ui}
    (async()=>{
      const node={id:7,imgs:[{src:location.origin+'/view?filename=test.png&subfolder=nested&type=output&metadata=1&id=7&rand=123'}]};
      const menu=extension.getNodeMenuItems(node);
      const valid=menu.length;
      node.imgs.push(node.imgs[0]); const multi=extension.getNodeMenuItems(node).length; node.imgs.pop();
      node.imgs[0].src+='&unknown=1'; const bad=extension.getNodeMenuItems(node).length; node.imgs[0].src=node.imgs[0].src.replace('&unknown=1','');
      menu[0].callback();
      let dialog=document.querySelector('dialog'); while(dialog.querySelector('button').disabled) await new Promise(r=>setTimeout(r,10)); const noSelectors=dialog.querySelectorAll('select,textarea').length;
      const buttons=dialog.querySelectorAll('button'); buttons[0].click(); buttons[0].click();
      const during={count:calls.length,sendDisabled:buttons[0].disabled,cancelDisabled:buttons[1].disabled};
      dialog.dispatchEvent(new Event('cancel',{cancelable:true})); const openDuring=dialog.open;
      resolveSend({ok:true,json:async()=>({status:'unknown'})}); await new Promise(r=>setTimeout(r,20));
      const status=dialog.querySelector('[role=status]').textContent;
      const payload=JSON.parse(calls[0].opts.body); buttons[0].click(); const after=calls.length;
      buttons[1].click();
      menu[0].callback(); dialog=document.querySelector('dialog[open]'); while(dialog.querySelector('button').disabled) await new Promise(r=>setTimeout(r,10)); node.imgs[0].src=location.origin+'/view?filename=other.png&type=output';
      dialog.querySelector('button').click(); const changed=dialog.querySelector('[role=status]').textContent;
      dialog.querySelectorAll('button')[1].click(); node.imgs[0].src=location.origin+'/view?filename=test.png&subfolder=nested&type=output';
      menu[0].callback(); dialog=document.querySelector('dialog[open]'); while(dialog.querySelector('button').disabled) await new Promise(r=>setTimeout(r,10));
      crypto.getRandomValues=()=>{throw new Error('entropy unavailable')}; dialog.querySelector('button').click();
      const entropyFailure=dialog.querySelector('[role=status]').textContent, finalCount=calls.length;
      document.body.dataset.result=JSON.stringify({valid,multi,bad,noSelectors,during,openDuring,status,payload,after,changed,entropyFailure,finalCount,url:calls[0].url});
    })().catch(e=>document.body.dataset.error=String(e));`;
  const server=createServer((req,res)=>{if(req.url==='/preview'){res.writeHead(200,{'content-type':'image/png','X-Image-SHA256':'a'.repeat(64)});res.end(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/aXcAAAAASUVORK5CYII=','base64'));return;} res.writeHead(200,{'content-type':'text/html; charset=utf-8'});res.end(`<body><script>${script}</script></body>`)});
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  try {
    const {stdout}=await exec('chromium',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--virtual-time-budget=3000','--dump-dom',`http://127.0.0.1:${server.address().port}/`],{timeout:30000,maxBuffer:2000000,env:process.env});
    const match=stdout.match(/data-result="([^"]+)"/); assert.ok(match,stdout.slice(0,2000));
    const r=JSON.parse(match[1].replaceAll('&quot;','"').replaceAll('&amp;','&'));
    assert.equal(r.valid,1); assert.equal(r.multi,0); assert.equal(r.bad,0); assert.equal(r.noSelectors,0);
    assert.deepEqual(r.during,{count:1,sendDisabled:true,cancelDisabled:true}); assert.equal(r.openDuring,true);
    assert.match(r.status,/结果未知/); assert.equal(r.after,1); assert.match(r.changed,/图片已变化/);
    assert.match(r.entropyFailure,/安全请求 ID 不可用/); assert.equal(r.finalCount,1);
    assert.equal(r.url,'/telegram_sender/image_context'); assert.deepEqual(Object.keys(r.payload).sort(),['filename','idempotency_key','sha256','subfolder','type']);
    assert.deepEqual({filename:r.payload.filename,subfolder:r.payload.subfolder,type:r.payload.type},{filename:'test.png',subfolder:'nested',type:'output'}); assert.match(r.payload.idempotency_key,/^[0-9a-f-]{36}$/); assert.equal(r.payload.sha256,'a'.repeat(64));
  } finally {server.close()}
});
