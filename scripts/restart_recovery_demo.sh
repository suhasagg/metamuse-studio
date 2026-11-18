set -e
start(){ MUSE_DATA_DIR=/tmp/muse-recovery-demo PYTHONPATH=. .venv/bin/uvicorn app.main:app --port 8011 >/tmp/srv.log 2>&1 & echo $! >/tmp/srv.pid; sleep 4; }
start
RID=$(curl -s -XPOST localhost:8011/api/runs -H 'content-type: application/json' -d '{"goal":"Plan a 3-day team offsite in Lisbon: research it, book flights and hotel, email the team","workflow":"trip","session_id":"restart"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
sleep 6; curl -s localhost:8011/api/runs/$RID | python3 -c "import sys,json;d=json.load(sys.stdin);print('before kill:',d['status'])"
kill -9 $(cat /tmp/srv.pid); sleep 1; start
curl -s localhost:8011/api/runs/$RID | python3 -c "import sys,json;d=json.load(sys.stdin);print('after restart:',d['status'],len(d['events']),'events');open('/tmp/pend.json','w').write(json.dumps({'decision':'approve','approved':d['pending_approval']['pending']}))"
curl -s -XPOST localhost:8011/api/runs/$RID/approval -H 'content-type: application/json' -d @/tmp/pend.json >/dev/null
sleep 8; curl -s localhost:8011/api/runs/$RID | python3 -c "import sys,json;d=json.load(sys.stdin);print('after approve:',d['status'],d['muse_score'].get('total'),d['error'])"
# mid-flight kill
RID2=$(curl -s -XPOST localhost:8011/api/runs -H 'content-type: application/json' -d '{"goal":"Launch Lumen, an AI notes app: research the market and write the campaign","workflow":"launch","session_id":"restart"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
sleep 1.5; kill -9 $(cat /tmp/srv.pid); sleep 1; start; sleep 10
curl -s localhost:8011/api/runs/$RID2 | python3 -c "import sys,json;d=json.load(sys.stdin);print('mid-flight:',d['status'],d['muse_score'].get('total'),[e for e in d['events'] if e['type']=='recovered'],d['error'])"
curl -s localhost:8011/api/security | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['audit'],d['recovery'])"
kill $(cat /tmp/srv.pid)
