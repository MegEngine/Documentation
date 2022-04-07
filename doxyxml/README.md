This folder is used to save the doxygen generated XML files for generate C++ API in Sphinx projects.

Make sure you have installed Sphinx, breathe and doxygen in your environment.

You can edit the `INPUT` tag setting in the `lite.config` file to specify lite directorie.
But it is suggested to create an symbolic link as the `INPUT` tag instead of editing it directly:

```shell
ln -s  /path/to/{MegEngine/lite/include/lite} /data/path-to-lite
```

Now you can run this commoand to generate/update MegEngine Lite C++ API xml files:

```shell
doxygen lite.config
```

See `source/reference/lite` for more detail and using examples.

