const sharp = require("sharp");

async function main() {
  const [source, destination, dpi] = process.argv.slice(2);
  await sharp(source, { density: 72 })
    .flatten({ background: "#ffffff" })
    .withMetadata({ density: Number(dpi) })
    .png({ compressionLevel: 9, adaptiveFiltering: false })
    .toFile(destination);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
