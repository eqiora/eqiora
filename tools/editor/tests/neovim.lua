local function run()
  local root = assert(vim.env.EQIORA_CLIENT_WORKSPACE)
  local source = vim.fn.readfile(assert(vim.env.EQIORA_CLIENT_SOURCE))
  for index, line in ipairs(source) do source[index] = line:gsub('rate: 1 / s', 'rate: Rate') end
  table.insert(source, 1, 'dimension Rate = 1 / s;')
  vim.fn.mkdir(root, 'p')
  vim.fn.writefile(source, root .. '/main.eqi')
  vim.env.PATH = vim.fn.fnamemodify(assert(vim.env.EQIORA_CLIENT_SERVER), ':h') .. ':' .. vim.env.PATH
  local guide = table.concat(vim.fn.readfile(assert(vim.env.EQIORA_CLIENT_GUIDE)), '\n')
  local config = assert(guide:match('```lua\n(.-)\n```'), 'documented Neovim configuration')
  local publications = {}
  local publish = vim.lsp.handlers['textDocument/publishDiagnostics']
  vim.lsp.handlers['textDocument/publishDiagnostics'] = function(error, result, context, options)
    assert(not error)
    table.insert(publications, result)
    publish(error, result, context, options)
  end
  vim.cmd('filetype plugin on') -- -u NONE disables normal startup detection.
  assert(loadstring(config))()
  vim.cmd.edit(root .. '/main.eqi')
  local buffer = vim.api.nvim_get_current_buf()
  assert(vim.bo.filetype == 'eqiora', 'documented filetype association')
  assert(vim.wait(10000, function()
    return #vim.lsp.get_clients({bufnr = buffer, name = 'eqiora'}) > 0
  end), 'documented client startup')
  local client = assert(vim.lsp.get_clients({bufnr = buffer, name = 'eqiora'})[1])
  assert(client.config.root_dir == root, 'documented root fallback')
  assert(vim.wait(10000, function() return #publications > 0 end), 'initial diagnostics')
  assert(#publications[#publications].diagnostics == 0, vim.inspect(publications))
  local document = { uri = vim.uri_from_bufnr(buffer) }
  local function request(method, params)
    local response = assert(client:request_sync(method, params, 10000, buffer), method)
    assert(not response.err, vim.inspect(response.err))
    return response.result
  end
  local symbols = request('textDocument/documentSymbol', {textDocument = document})
  assert(symbols[1].name == 'Rate' and symbols[2].name == 'decay', vim.inspect(symbols))
  local reference
  for index, line in ipairs(source) do
    local start = line:find('Rate = 1;', 1, true)
    if start then reference = {line = index - 1, character = start - 1} end
  end
  local at = {textDocument = document, position = assert(reference)}
  local hover = request('textDocument/hover', at)
  assert(hover and vim.inspect(hover):find('Rate', 1, true), vim.inspect(hover))
  local locations = request('textDocument/definition', at)
  local location = locations.uri and locations or locations[1]
  assert(location.uri == document.uri, vim.inspect(locations))
  assert(source[location.range.start.line + 1]:find('dimension Rate', 1, true))
  local edits = request('textDocument/formatting', {
    textDocument = document, options = {tabSize = 2, insertSpaces = true},
  })
  assert(#edits > 0)
  vim.lsp.util.apply_text_edits(edits, buffer, client.offset_encoding)
  local function replace(lines, expect_error)
    local count = #publications
    vim.api.nvim_buf_set_lines(buffer, 0, -1, false, lines)
    assert(vim.wait(10000, function()
      return #publications > count and publications[#publications].version == vim.lsp.util.buf_versions[buffer]
    end), 'changed diagnostics')
    local diagnostics = publications[#publications].diagnostics
    assert((#diagnostics > 0) == expect_error, vim.inspect(diagnostics))
    return diagnostics
  end
  local invalid = {}
  for index, line in ipairs(source) do invalid[index] = line:gsub('Rate = 1 / s', 'Rate = m') end
  local errors = replace(invalid, true)
  assert(vim.inspect(errors):find('dimension', 1, true), vim.inspect(errors))
  replace(source, false)
  vim.api.nvim_buf_delete(buffer, {force = true})
  client:stop()
  assert(vim.wait(10000, function() return client:is_stopped() end), 'shutdown')
  print('Neovim: open, diagnostics, hover, symbols, definition, format, unit error, recovery, close passed')
end
local ok, error = xpcall(run, debug.traceback)
if not ok then
  io.stderr:write(error .. '\n')
  vim.cmd.cquit()
end
vim.cmd('qa!')
