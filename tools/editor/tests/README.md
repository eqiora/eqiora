# Released-server client checks

`neovim.lua` and `vim.vim` execute the configuration blocks from the canonical
[editor guide](../../../docs/site/src/content/docs/guides/editors.mdx). They use
`examples/decay.eqi`, introducing a dimension alias to exercise supported
navigation and a deliberate unit mismatch. Both check hover, symbols,
definition, formatting, diagnostics recovery and shutdown. They require the
released server and editor/client versions identified in that guide; downloading
or installing tools is deliberately separate from the tests.

From the repository root, point `EQIORA_CLIENT_SERVER` at the extracted released
server and put the tested Neovim on PATH. Use separate disposable workspaces:

```bash
export EQIORA_CLIENT_SOURCE="$PWD/examples/decay.eqi"
export EQIORA_CLIENT_GUIDE="$PWD/docs/site/src/content/docs/guides/editors.mdx"
export EQIORA_CLIENT_SERVER="$PWD/.tools/eqiora-0.1.1/eqiora-language-server"
export EQIORA_CLIENT_WORKSPACE="$HOME/.cache/eqiora/editor-check/neovim"
nvim --headless -u NONE -l tools/editor/tests/neovim.lua

export EQIORA_VIM_LSP="$HOME/.vim/pack/eqiora/start/vim-lsp"
export EQIORA_CLIENT_WORKSPACE="$HOME/.cache/eqiora/editor-check/vim"
vim -Nu NONE -n -es -S tools/editor/tests/vim.vim
cat "$EQIORA_CLIENT_WORKSPACE/result.txt"
```

Each command exits nonzero on a failed assertion. Vim additionally writes the
failure and call location to `result.txt`; its LSP log stays in the same scratch
directory. The Neovim harness enables normal filetype startup because `-u NONE`
disables it; neither harness substitutes a server registration for the guide's
configuration. These are focused client tests, not a registered scientific
claim or a promise about untested platforms and remote setups.
