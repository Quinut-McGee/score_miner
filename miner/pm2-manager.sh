#!/bin/bash
# PM2 Manager for Score Vision Miner

MINER_NAME="sn44-miner"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$1" in
    start)
        echo "Starting Score Vision Miner with optimized settings..."
        pm2 start ecosystem.config.js
        pm2 save
        ;;
    stop)
        echo "Stopping Score Vision Miner..."
        pm2 stop $MINER_NAME
        ;;
    restart)
        echo "Restarting Score Vision Miner with optimized settings..."
        pm2 restart $MINER_NAME
        ;;
    reload)
        echo "Reloading Score Vision Miner..."
        pm2 reload $MINER_NAME
        ;;
    delete)
        echo "Deleting Score Vision Miner from PM2..."
        pm2 delete $MINER_NAME
        ;;
    status)
        echo "Score Vision Miner Status:"
        pm2 status $MINER_NAME
        ;;
    logs)
        echo "Showing Score Vision Miner logs:"
        pm2 logs $MINER_NAME --lines 50
        ;;
    monitor)
        echo "Opening PM2 monitor..."
        pm2 monit
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|reload|delete|status|logs|monitor}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the miner with optimized settings"
        echo "  stop    - Stop the miner"
        echo "  restart - Restart the miner with optimized settings"
        echo "  reload  - Reload the miner (zero-downtime)"
        echo "  delete  - Remove the miner from PM2"
        echo "  status  - Show miner status"
        echo "  logs    - Show recent logs"
        echo "  monitor - Open PM2 monitoring interface"
        echo ""
        echo "Optimized Configuration:"
        echo "  Batch Size: 4"
        echo "  Expected FPS: 37.4 (vs 21 previously)"
        echo "  GPU: RTX 5070 Ti (15.5 GB)"
        exit 1
        ;;
esac
