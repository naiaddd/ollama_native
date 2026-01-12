#!/bin/bash
# Clear the database and attachments folder
rm -f history.db
rm -rf ./attachments/*
echo "Application state reset: history.db and attachments cleared."
