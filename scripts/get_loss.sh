#!/bin/bash
grep "Best" $1 | grep shape | cut -d' ' -f10,15 > $2
