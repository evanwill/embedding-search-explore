require "fileutils"
require "shellwords"

PROJECT_ROOT = File.expand_path("..", __dir__)
SCRIPTS_DIR = File.join(PROJECT_ROOT, "embeddings", "scripts")
DATA_DIR = File.join(PROJECT_ROOT, "assets", "embeddings", "data")
LIB_DIR = File.join(PROJECT_ROOT, "assets", "embeddings", "lib")
NODE_MODULES = File.join(SCRIPTS_DIR, "node_modules")
MIN_NODE_MAJOR = 18

# Browser runtime files vendored from the pinned npm install so the search
# page runs the exact library version the build used, with no CDN dependency.
# transformers.min.js is the self-contained browser bundle (the .web.* builds
# have bare onnxruntime imports and need a bundler). The .wasm/.mjs pair is
# the ONNX runtime (covers both WASM and WebGPU).
VENDOR_DIST_DIR = File.join(NODE_MODULES, "@huggingface", "transformers", "dist")
VENDOR_FILES = %w[
  transformers.min.js
  ort-wasm-simd-threaded.jsep.mjs
  ort-wasm-simd-threaded.jsep.wasm
].freeze


# Cross-platform executable lookup (command -v is POSIX-only).
def find_executable(name)
  extensions = Gem.win_platform? ? %w[.exe .cmd .bat] : [""]
  ENV.fetch("PATH", "").split(File::PATH_SEPARATOR).each do |dir|
    extensions.each do |ext|
      candidate = File.join(dir, "#{name}#{ext}")
      return candidate if File.file?(candidate) && File.executable?(candidate)
    end
  end
  nil
end


def node_version(node_cmd)
  version = `#{Shellwords.escape(node_cmd)} --version`.strip
  version[/\d+/]&.to_i
rescue StandardError
  nil
end


def find_node!
  node_cmd = find_executable("node")
  unless node_cmd
    abort(
      "Node.js #{MIN_NODE_MAJOR}+ is required to build embeddings.\n" \
      "Install the LTS release from https://nodejs.org/ then run `rake setup` again."
    )
  end

  major = node_version(node_cmd)
  if major.nil? || major < MIN_NODE_MAJOR
    abort(
      "Node.js #{MIN_NODE_MAJOR}+ is required (found #{major || 'unknown'}).\n" \
      "Install the LTS release from https://nodejs.org/ then run `rake setup` again."
    )
  end

  node_cmd
end


# Prefer pnpm when its lockfile is present (this repo ships pnpm-lock.yaml);
# fall back to npm, which comes bundled with Node.
def find_package_manager!
  if File.exist?(File.join(SCRIPTS_DIR, "pnpm-lock.yaml")) && find_executable("pnpm")
    return "pnpm"
  end
  return "npm" if find_executable("npm")
  return "pnpm" if find_executable("pnpm")

  abort(
    "Neither npm nor pnpm was found on PATH.\n" \
    "npm is bundled with Node.js: https://nodejs.org/"
  )
end


desc "Check Node.js and install JavaScript dependencies"
task :setup_embeddings do
  find_node!
  package_manager = find_package_manager!

  puts "Installing JavaScript dependencies with #{package_manager} ..."
  Dir.chdir(SCRIPTS_DIR) do
    sh("#{package_manager} install")
  end

  puts "Setup complete."
end


desc "Copy the pinned Transformers.js browser runtime into assets/embeddings/lib"
task :vendor_embeddings_lib do
  Rake::Task[:setup_embeddings].invoke unless File.directory?(NODE_MODULES)

  missing = VENDOR_FILES.reject { |name| File.exist?(File.join(VENDOR_DIST_DIR, name)) }
  unless missing.empty?
    abort(
      "Missing runtime file(s) in #{VENDOR_DIST_DIR}: #{missing.join(', ')}.\n" \
      "Re-run `rake setup_embeddings`; if the package layout changed, update VENDOR_FILES."
    )
  end

  FileUtils.mkdir_p(LIB_DIR)
  VENDOR_FILES.each do |name|
    FileUtils.cp(File.join(VENDOR_DIST_DIR, name), File.join(LIB_DIR, name))
  end
  puts "Vendored #{VENDOR_FILES.length} runtime file(s) into #{LIB_DIR}."
end


desc "Generate embedding artifacts for browser search (edit embeddings/config-embeddings.yml first)"
task :build_embeddings do
  Rake::Task[:setup_embeddings].invoke unless File.directory?(NODE_MODULES)
  Rake::Task[:vendor_embeddings_lib].invoke

  node_cmd = find_node!
  build_script = File.join(SCRIPTS_DIR, "build_embeddings.mjs")
  abort("Missing build script: #{build_script}") unless File.exist?(build_script)

  puts "Running preprocessing (first run downloads model weights) ..."
  Dir.chdir(SCRIPTS_DIR) do
    sh("#{Shellwords.escape(node_cmd)} build_embeddings.mjs")
  end

  puts "Embedding build complete."
end


desc "Remove generated data artifacts"
task :clean_data do
  artifacts = %w[manifest.json embeddings.bin index.json build-info.json preprocess.log]
    .map { |name| File.join(DATA_DIR, name) }

  existing = artifacts.select { |path| File.exist?(path) }
  if existing.empty?
    puts "No generated artifacts found in #{DATA_DIR}."
  else
    FileUtils.rm_f(existing)
    puts "Removed #{existing.length} generated artifact(s)."
  end
end
