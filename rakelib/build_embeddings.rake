require "fileutils"
require "shellwords"

PROJECT_ROOT = File.expand_path(__dir__)
SCRIPTS_DIR = File.join(PROJECT_ROOT, "embeddings")
DATA_DIR = File.join(PROJECT_ROOT, "assets", "embeddings", "data")
NODE_MODULES = File.join(SCRIPTS_DIR, "node_modules")
MIN_NODE_MAJOR = 18


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


desc "Generate embedding artifacts for browser search (edit config.yml first)"
task :build_embeddings do
  Rake::Task[:setup].invoke unless File.directory?(NODE_MODULES)

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
