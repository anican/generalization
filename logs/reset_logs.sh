#!/usr/bin/env bash

# WARNING!!! Deletes all logs
read -p "Delete existing checkpoints and log files (y/n)?" choice
case "$choice" in
  y|Y )
	echo "Deleting existing checkpoints and log files..."
	rm -rf gradient_tape/
	echo "Done!"
	;;
  n|N )
	echo "no"
	;;
  * )
	echo "invalid"
	;;
esac


