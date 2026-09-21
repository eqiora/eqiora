set nocompatible
set nomore
execute 'set runtimepath^=' . fnameescape($EQIORA_VIM_LSP)
let g:lsp_use_native_client = 1
let g:lsp_log_verbose = 1
let g:lsp_log_file = $EQIORA_CLIENT_WORKSPACE . '/vim-lsp.log'
call mkdir($EQIORA_CLIENT_WORKSPACE, 'p')
function! s:Wait(Fn, label) abort
  for attempt in range(1000)
    if a:Fn() | return | endif
    sleep 10m
  endfor
  throw 'timed out: ' . a:label
endfunction
function! s:Response(data) abort
  let g:response = a:data
endfunction
function! s:Request(method, params) abort
  unlet! g:response
  call lsp#send_request('eqiora', {
        \ 'method': a:method, 'params': a:params,
        \ 'on_notification': function('s:Response')})
  call s:Wait({-> exists('g:response')}, a:method)
  if !has_key(g:response, 'response') || has_key(g:response.response, 'error')
    throw string(g:response)
  endif
  return g:response.response.result
endfunction
try
  let $PATH = fnamemodify($EQIORA_CLIENT_SERVER, ':h') . ':' . $PATH
  let s:guide = readfile($EQIORA_CLIENT_GUIDE)
  let s:start = index(s:guide, '```vim') + 1
  call assert_true(s:start > 0)
  let s:end = index(s:guide, '```', s:start)
  let s:config = $EQIORA_CLIENT_WORKSPACE . '/documented.vim'
  call writefile(s:guide[s:start : s:end - 1], s:config)
  execute 'source ' . fnameescape(s:config)
  runtime plugin/lsp.vim
  let s:source = readfile($EQIORA_CLIENT_SOURCE)
  call map(s:source, {_, line->substitute(line, 'rate: 1 / s', 'rate: Rate', '')})
  call insert(s:source, 'dimension Rate = 1 / s;')
  call writefile(s:source, $EQIORA_CLIENT_WORKSPACE . '/main.eqi')
  execute 'edit ' . fnameescape($EQIORA_CLIENT_WORKSPACE . '/main.eqi')
  call assert_equal('eqiora', &filetype)
  call lsp#enable()
  let s:document = {'uri': lsp#utils#get_buffer_uri()}
  let s:symbols = s:Request('textDocument/documentSymbol', {'textDocument': s:document})
  call assert_equal(lsp#utils#path_to_uri($EQIORA_CLIENT_WORKSPACE), lsp#get_server_root_uri('eqiora'))
  call assert_equal('Rate', s:symbols[0].name)
  call assert_equal('decay', s:symbols[1].name)
  for index in range(len(s:source))
    let column = stridx(s:source[index], 'Rate = 1;')
    if column >= 0 | let s:at = {'textDocument': s:document, 'position': {'line': index, 'character': column}} | endif
  endfor
  call assert_match('Rate', string(s:Request('textDocument/hover', s:at)))
  let s:definition = s:Request('textDocument/definition', s:at)
  call assert_equal(s:document.uri, s:definition.uri)
  call assert_equal(0, s:definition.range.start.line)
  let s:formatted = s:Request('textDocument/formatting', {'textDocument': s:document, 'options': {'tabSize': 2, 'insertSpaces': v:true}})
  call assert_true(len(s:formatted) > 0)
  LspDocumentFormatSync
  call assert_equal(split(s:formatted[0].newText, "\n"), getline(1, '$'))
  let s:invalid = map(copy(s:source), {_, line->substitute(line, 'Rate = 1 / s', 'Rate = m', '')})
  call setline(1, s:invalid)
  if line('$') > len(s:invalid) | execute (len(s:invalid) + 1) . ',$delete _' | endif
  call s:Request('textDocument/documentSymbol', {'textDocument': s:document})
  call s:Wait({->lsp#get_buffer_diagnostics_counts().error > 0}, 'unit mismatch')
  call setline(1, s:source)
  call s:Request('textDocument/documentSymbol', {'textDocument': s:document})
  call s:Wait({->lsp#get_buffer_diagnostics_counts().error == 0}, 'corrected dimensions')
  bwipeout!
  call lsp#stop_server('eqiora')
  if !empty(v:errors) | throw join(v:errors, '\n') | endif
  call writefile(['Vim: open, hover, symbols, definition, format, unit error, recovery, close passed'], $EQIORA_CLIENT_WORKSPACE . '/result.txt')
catch
  call writefile([v:exception, v:throwpoint] + v:errors, $EQIORA_CLIENT_WORKSPACE . '/result.txt')
  cquit
endtry
qa!
