#!/bin/bash -e

make BUILDDIR=/tmp LANGUAGE="zh_CN" html
mv /tmp/html zh
make BUILDDIR=/tmp LANGUAGE="en" html
mv /tmp/html en
tar czf static.tgz zh en
rm -r zh en

