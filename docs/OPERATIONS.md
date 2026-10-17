# Operations and rollback

Default: one voicey-bot-1 container, own data, read-only models/config, no public
ports. keykomi has 1 vCPU and 1963 MiB RAM. Existing VPN/gateway services are protected:
do not recreate their containers, volumes, firewall, Docker daemon or reverse proxy.

Before activation record Git revision/image ID, container list, free memory, swap,
free disk and failed units. For updates stop the old bot cleanly, back up its DB,
save the old image and compatible config; never run two pollers during migration.

```sh
docker compose ps
docker compose logs --tail=30 bot
docker stats --no-stream voicey-bot-1
docker compose stop bot
docker compose up -d --no-deps bot
```

Never print expanded `docker compose config` or docker inspect environment values.
Logs contain metadata. Results saved before failed Telegram delivery can be recovered
locally from jobs; do not dump private text into shared logs.

Back up online using SQLite Connection.backup(), or stop the bot and copy all of
data/ including WAL/SHM. Protect backups like private messages. Preserve `.env`,
operator profiles and checksum manifest separately; never commit them to Git.

Health gates: no restart loop; expected bot identity; allowed-user /start/settings;
complete transcript replying to its voice; separate text-action reply; model child
PIDs disappear afterwards; no OOM, sustained swap I/O or VPN degradation. Observe
idle and one job per model; record RAM and audio-duration/wall-time ratio. API
providers require separate live credentials/tests.

Initial rollback: `docker compose stop bot`, keeping models, data and configuration.
Update rollback: restore previous image and compatible configuration, start one
instance. No destructive schema migrations currently exist. Restore a complete DB
backup only while stopped. Target bot rollback time: under one minute.

If new host swap was configured, first stop the bot and assess RAM before swapoff.
Disable only the added zswap unit/config, then swapoff the exact added file and
remove its fstab entry. Never blanket-swapoff or remove pre-existing devices.
Swap rollback may take longer under pressure and is separate from application rollback.

`docker stats`, process RSS and cgroup memory peak differ (file-cache accounting).
Record what was actually measured. RTF = processing seconds / audio seconds.
Weight size and author GPU speed are not evidence of VPS runtime memory or latency.
